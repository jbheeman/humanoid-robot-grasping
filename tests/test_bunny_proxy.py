from g1_bunny_vla.bunny_proxy import DEFAULT_BUNNY_PROXY


def test_measured_bunny_proxy_specification_is_valid():
    DEFAULT_BUNNY_PROXY.validate()
    assert DEFAULT_BUNNY_PROXY.height_m == 0.14
    assert DEFAULT_BUNNY_PROXY.depth_m == 0.15
    assert DEFAULT_BUNNY_PROXY.width_m == 0.07
    assert DEFAULT_BUNNY_PROXY.mass_kg == 0.15
    assert DEFAULT_BUNNY_PROXY.max_push_speed_mps <= 0.15
