$ver = Get-Content ver.txt
$ver = ([int]$ver + 1)
Write-Output $ver

Write-Output "Preparing nerw.js for distribution..."
Write-Output "Creating nerw-docs.js file"
terser --compress --mangle --comments some nerw.js -o "dist/nerw-docs-v$ver.js"
Write-Output "Created nerw-docs.js file"
Write-Output "Creating nerw.js file"
terser --compress --mangle --comments false nerw.js -o "dist/nerw-nohead-v$ver.js"
Write-Output "Created nerw.js file"
# Start-Sleep -Milliseconds 2000
Write-Output "Adding header to compressed dist"
Get-Content header.txt, "dist/nerw-nohead-v$ver.js" | Set-Content "dist/nerw-v$ver.js"
Write-Output "Added header to compressed dist"
Write-Output "nerw.js dist files writted in dist folder"

[String]$ver | Out-File -FilePath ver.txt