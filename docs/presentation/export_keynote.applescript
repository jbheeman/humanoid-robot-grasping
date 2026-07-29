on run argv
	if (count of argv) is not 3 then error "Usage: export_keynote.applescript input.pptx output.key output.pdf"
	set inputPath to item 1 of argv
	set keynotePath to item 2 of argv
	set pdfPath to item 3 of argv
	set inputFile to POSIX file inputPath as alias
	set keynoteFile to POSIX file keynotePath
	set pdfFile to POSIX file pdfPath
	
	tell application "Keynote"
		activate
		open inputFile
		delay 8
		set deckDocument to front document
		save deckDocument in keynoteFile
		export deckDocument to pdfFile as PDF
		close deckDocument saving no
	end tell
end run
