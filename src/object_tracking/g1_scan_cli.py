from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from time import monotonic


DEFAULT_PORTS = (22, 80, 443, 554, 8080, 8081, 8554, 50051, 50052)
UNITREE_MAC_HINTS = (
    "Unitree",
    "Hangzhou Yushu",
    "Yushu Technology",
)


@dataclass
class InterfaceNetwork:
    interface: str
    network: ipaddress.IPv4Network
    address: str
    default_route: bool = False


@dataclass
class Candidate:
    ip: str
    alive: bool = False
    hostname: str | None = None
    mac: str | None = None
    vendor: str | None = None
    open_ports: list[int] = field(default_factory=list)
    score: int = 0
    reasons: list[str] = field(default_factory=list)


def run_json(command: list[str], timeout: float = 3.0) -> object | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def default_route_interface() -> str | None:
    data = run_json(["ip", "-j", "route", "show", "default"])
    if not isinstance(data, list):
        return None
    for route in data:
        if isinstance(route, dict) and route.get("dev"):
            return str(route["dev"])
    return None


def local_networks(preferred_interface: str | None, include_all: bool) -> list[InterfaceNetwork]:
    default_iface = default_route_interface()
    data = run_json(["ip", "-j", "-4", "addr", "show", "scope", "global"])
    networks: list[InterfaceNetwork] = []
    if not isinstance(data, list):
        return networks

    for iface in data:
        if not isinstance(iface, dict):
            continue
        name = str(iface.get("ifname", ""))
        if preferred_interface and name != preferred_interface:
            continue
        if not include_all and preferred_interface is None and default_iface and name != default_iface:
            continue
        if name.startswith(("docker", "br-", "veth", "virbr")):
            continue
        for addr in iface.get("addr_info", []):
            if not isinstance(addr, dict) or addr.get("family") != "inet":
                continue
            local = addr.get("local")
            prefix = addr.get("prefixlen")
            if local is None or prefix is None:
                continue
            network = ipaddress.ip_network(f"{local}/{prefix}", strict=False)
            if network.prefixlen < 16:
                continue
            networks.append(
                InterfaceNetwork(
                    interface=name,
                    network=network,
                    address=str(local),
                    default_route=name == default_iface,
                )
            )
    return networks


def limited_hosts(network: ipaddress.IPv4Network, limit: int | None) -> list[str]:
    hosts = [str(host) for host in network.hosts()]
    if limit is not None and len(hosts) > limit:
        raise SystemExit(
            f"Refusing to scan {network} ({len(hosts)} hosts). "
            f"Pass --limit-hosts {len(hosts)} or choose a tighter --cidr."
        )
    return hosts


async def ping_host(ip: str, timeout_s: float) -> bool:
    timeout_text = str(max(1, int(timeout_s)))
    command = ["ping", "-n", "-c", "1", "-W", timeout_text, ip]
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await proc.wait() == 0
    except OSError:
        return False


async def probe_port(ip: str, port: int, timeout_s: float) -> bool:
    try:
        _reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port),
            timeout=timeout_s,
        )
    except (OSError, asyncio.TimeoutError):
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    return True


async def bounded_map(items: list[str], limit: int, worker):
    semaphore = asyncio.Semaphore(limit)

    async def run_one(item):
        async with semaphore:
            return item, await worker(item)

    return await asyncio.gather(*(run_one(item) for item in items))


def parse_neigh() -> dict[str, dict[str, str]]:
    data = run_json(["ip", "-j", "neigh"], timeout=3.0)
    result: dict[str, dict[str, str]] = {}
    if not isinstance(data, list):
        return result
    for row in data:
        if not isinstance(row, dict):
            continue
        dst = row.get("dst")
        if not dst:
            continue
        entry: dict[str, str] = {}
        if row.get("lladdr"):
            entry["mac"] = str(row["lladdr"])
        if row.get("dev"):
            entry["interface"] = str(row["dev"])
        if row.get("state"):
            entry["state"] = str(row["state"])
        result[str(dst)] = entry
    return result


def nmap_vendor_scan(ips: list[str], timeout_s: float) -> dict[str, dict[str, str]]:
    if not ips or shutil.which("nmap") is None:
        return {}
    command = ["nmap", "-sn", "-n", *ips]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(timeout_s, 5.0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    vendors: dict[str, dict[str, str]] = {}
    current_ip: str | None = None
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line.startswith("Nmap scan report for "):
            current_ip = line.rsplit(" ", 1)[-1]
        elif current_ip and line.startswith("MAC Address: "):
            rest = line.removeprefix("MAC Address: ").strip()
            mac, _, vendor = rest.partition(" ")
            vendors[current_ip] = {
                "mac": mac,
                "vendor": vendor.strip("()") if vendor else "",
            }
    return vendors


def reverse_hostname(ip: str, timeout_s: float) -> str | None:
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)
    try:
        return socket.gethostbyaddr(ip)[0]
    except (OSError, socket.herror):
        return None
    finally:
        socket.setdefaulttimeout(old_timeout)


def score_candidate(candidate: Candidate) -> None:
    if candidate.alive:
        candidate.score += 10
        candidate.reasons.append("responded to ping")
    if candidate.mac:
        candidate.score += 5
        candidate.reasons.append(f"has ARP MAC {candidate.mac}")
    if candidate.vendor:
        candidate.score += 8
        candidate.reasons.append(f"vendor: {candidate.vendor}")
        if any(hint.lower() in candidate.vendor.lower() for hint in UNITREE_MAC_HINTS):
            candidate.score += 50
            candidate.reasons.append("vendor matches Unitree hint")
    if candidate.hostname:
        candidate.score += 4
        candidate.reasons.append(f"hostname: {candidate.hostname}")
        if any(word in candidate.hostname.lower() for word in ("unitree", "g1", "robot")):
            candidate.score += 25
            candidate.reasons.append("hostname looks robot-related")
    if 22 in candidate.open_ports:
        candidate.score += 12
        candidate.reasons.append("SSH open")
    if 554 in candidate.open_ports or 8554 in candidate.open_ports:
        candidate.score += 8
        candidate.reasons.append("RTSP-like port open")
    web_ports = [port for port in candidate.open_ports if port in (80, 443, 8080, 8081)]
    if web_ports:
        candidate.score += 4
        candidate.reasons.append(f"web-like port open: {','.join(str(port) for port in web_ports)}")
    if 50051 in candidate.open_ports or 50052 in candidate.open_ports:
        candidate.score += 4
        candidate.reasons.append("gRPC-like port open")


async def scan_hosts(
    hosts: list[str],
    ports: list[int],
    ping_timeout: float,
    connect_timeout: float,
    concurrency: int,
    include_sleeping: bool,
    resolve_names: bool,
    use_nmap: bool,
) -> list[Candidate]:
    start = monotonic()
    ping_results = await bounded_map(hosts, concurrency, lambda ip: ping_host(ip, ping_timeout))
    alive_hosts = [ip for ip, ok in ping_results if ok]

    neigh = parse_neigh()
    host_set = set(hosts)
    known_hosts = set(alive_hosts)
    known_hosts.update(ip for ip in neigh if ip in host_set)
    if include_sleeping:
        known_hosts.update(hosts)

    candidates = {ip: Candidate(ip=ip, alive=ip in alive_hosts) for ip in sorted(known_hosts, key=ipaddress.ip_address)}
    for ip, entry in neigh.items():
        candidate = candidates.get(ip)
        if candidate is None:
            continue
        candidate.mac = entry.get("mac")

    if use_nmap:
        for ip, info in nmap_vendor_scan(list(candidates), timeout_s=max(10.0, monotonic() - start + 5.0)).items():
            candidate = candidates.get(ip)
            if candidate is None:
                continue
            candidate.mac = candidate.mac or info.get("mac")
            candidate.vendor = info.get("vendor") or candidate.vendor

    async def scan_ports(ip: str) -> list[int]:
        checks = await asyncio.gather(*(probe_port(ip, port, connect_timeout) for port in ports))
        return [port for port, ok in zip(ports, checks) if ok]

    port_results = await bounded_map(list(candidates), concurrency, scan_ports)
    for ip, open_ports in port_results:
        candidates[ip].open_ports = open_ports

    if resolve_names:
        for candidate in candidates.values():
            candidate.hostname = reverse_hostname(candidate.ip, timeout_s=0.3)

    for candidate in candidates.values():
        score_candidate(candidate)

    return sorted(candidates.values(), key=lambda item: (-item.score, ipaddress.ip_address(item.ip)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find likely Unitree G1 robot IPs on the current LAN."
    )
    parser.add_argument("--cidr", help="Subnet to scan, for example 192.168.1.0/24.")
    parser.add_argument("--interface", help="Only scan the IPv4 subnet attached to this interface.")
    parser.add_argument(
        "--all-private",
        action="store_true",
        help="Scan every non-virtual global IPv4 interface instead of only the default route interface.",
    )
    parser.add_argument(
        "--ports",
        default=",".join(str(port) for port in DEFAULT_PORTS),
        help="Comma-separated TCP ports to probe after ping sweep.",
    )
    parser.add_argument("--ping-timeout", type=float, default=1.0, help="Seconds per ping attempt.")
    parser.add_argument("--connect-timeout", type=float, default=0.25, help="Seconds per TCP connect probe.")
    parser.add_argument("--concurrency", type=int, default=128, help="Concurrent ping/port probes.")
    parser.add_argument("--limit-hosts", type=int, default=512, help="Refuse broader scans unless raised.")
    parser.add_argument(
        "--include-sleeping",
        action="store_true",
        help="Also port-probe hosts that did not answer ping. Slower, but useful on networks blocking ICMP.",
    )
    parser.add_argument("--resolve-names", action="store_true", help="Try reverse DNS hostnames.")
    parser.add_argument(
        "--no-nmap",
        action="store_true",
        help="Do not use nmap for MAC vendor lookup even when nmap is installed.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def parse_ports(raw: str) -> list[int]:
    ports: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            port = int(part)
        except ValueError as exc:
            raise SystemExit(f"invalid port: {part}") from exc
        if port < 1 or port > 65535:
            raise argparse.ArgumentTypeError(f"invalid port: {port}")
        ports.append(port)
    return sorted(set(ports))


def resolve_scan_networks(args: argparse.Namespace) -> list[InterfaceNetwork]:
    if args.cidr:
        network = ipaddress.ip_network(args.cidr, strict=False)
        if not isinstance(network, ipaddress.IPv4Network):
            raise SystemExit("--cidr must be an IPv4 subnet")
        return [InterfaceNetwork(interface=args.interface or "manual", network=network, address="")]

    networks = local_networks(args.interface, args.all_private)
    if not networks:
        hint = " Check `ip -4 addr` or pass --cidr 192.168.x.0/24."
        raise SystemExit(f"Could not discover a LAN subnet to scan.{hint}")
    return networks


def print_text(networks: list[InterfaceNetwork], candidates: list[Candidate], elapsed_s: float) -> None:
    print("Unitree G1 LAN scan")
    print("Scanned:")
    for network in networks:
        marker = " default-route" if network.default_route else ""
        print(f"  - {network.network} on {network.interface}{marker}")
    print(f"Elapsed: {elapsed_s:.1f}s")
    print()

    if not candidates:
        print("No reachable candidates found.")
        print("Try: uv run g1-scan --include-sleeping --connect-timeout 0.5")
        return

    print("Likely candidates:")
    for index, candidate in enumerate(candidates[:12], start=1):
        ports = ",".join(str(port) for port in candidate.open_ports) or "-"
        identity = candidate.hostname or candidate.vendor or candidate.mac or "unknown"
        print(f"{index:>2}. {candidate.ip:<15} score={candidate.score:<3} ports={ports:<18} {identity}")
        if candidate.reasons:
            print(f"    {'; '.join(candidate.reasons[:5])}")

    best = candidates[0]
    if best.score > 0:
        print()
        print(f"Best guess: {best.ip}")
        print(f"Try: uv run loco {best.ip} --diagnose")
        print(f"Then: uv run vision {best.ip} --diagnose")


def main() -> None:
    args = build_parser().parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")

    ports = parse_ports(args.ports)
    networks = resolve_scan_networks(args)
    hosts: list[str] = []
    for network in networks:
        hosts.extend(limited_hosts(network.network, args.limit_hosts))
    hosts = sorted(set(hosts), key=ipaddress.ip_address)

    start = monotonic()
    candidates = asyncio.run(
        scan_hosts(
            hosts=hosts,
            ports=ports,
            ping_timeout=args.ping_timeout,
            connect_timeout=args.connect_timeout,
            concurrency=args.concurrency,
            include_sleeping=args.include_sleeping,
            resolve_names=args.resolve_names,
            use_nmap=not args.no_nmap,
        )
    )
    elapsed = monotonic() - start

    if args.json:
        payload = {
            "networks": [
                {
                    "interface": network.interface,
                    "network": str(network.network),
                    "address": network.address,
                    "default_route": network.default_route,
                }
                for network in networks
            ],
            "elapsed_s": round(elapsed, 3),
            "candidates": [
                {
                    "ip": candidate.ip,
                    "score": candidate.score,
                    "alive": candidate.alive,
                    "hostname": candidate.hostname,
                    "mac": candidate.mac,
                    "vendor": candidate.vendor,
                    "open_ports": candidate.open_ports,
                    "reasons": candidate.reasons,
                }
                for candidate in candidates
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print_text(networks, candidates, elapsed)


if __name__ == "__main__":
    main()
