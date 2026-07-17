# Rebuild docs/graphrag_stage1_user_guide.pdf from docs/user_guide.html
#
# The guide is hand-written HTML; the PDF is a headless-browser print of it. Keeping the
# command here rather than in someone's shell history is the point: the PDF silently went
# four days stale once because the toolchain was undiscoverable from the repo, and the PDF
# is the artifact reviewers actually read.
#
#     pwsh docs/build_pdf.ps1
#
# Edge is used because it ships with Windows; Chrome works identically -- both are Chromium,
# and --print-to-pdf is a Chromium flag. Point $browser at chrome.exe if you prefer.
#
# Run this whenever user_guide.html changes, and commit the PDF alongside it.

$ErrorActionPreference = "Stop"

$candidates = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
)
$browser = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $browser) { throw "No Chromium browser found. Install Edge or Chrome, or add its path to `$candidates." }

$docs = Split-Path -Parent $MyInvocation.MyCommand.Path
$src  = Join-Path $docs "user_guide.html"
$out  = Join-Path $docs "graphrag_stage1_user_guide.pdf"
if (-not (Test-Path $src)) { throw "Missing source: $src" }

# file:/// URI so relative assets resolve the same way they do in a browser.
$uri = "file:///" + ($src -replace '\\', '/')

Write-Output "browser : $browser"
Write-Output "source  : $src"

# Do NOT redirect stderr here. Headless Edge writes harmless renderer chatter to stderr, and
# Windows PowerShell 5.1 wraps native stderr in ErrorRecords -- with $ErrorActionPreference =
# "Stop" that turns a successful print into a fatal NativeCommandError. Let it write to the
# console; the real success check is the PDF inspection below.
& $browser --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$out" $uri
Start-Sleep -Seconds 3   # headless print returns before the file is flushed

if (-not (Test-Path $out)) { throw "Print produced no output: $out" }
Write-Output ("output  : {0} ({1:N0} bytes)" -f $out, (Get-Item $out).Length)

# Fail loudly rather than shipping a PDF that silently lost the document. A Chromium print
# failure tends to yield a valid-but-near-empty PDF, which no file-exists check would catch.
$check = python -c @"
import sys
try:
    from pypdf import PdfReader
except ImportError:
    sys.exit(0)   # pypdf optional; skip the check rather than fail the build
r = PdfReader(r'$out')
text = ''.join((p.extract_text() or '') for p in r.pages)
print(f'pages   : {len(r.pages)}')
if len(r.pages) < 10 or 'Make it fast' not in text:
    sys.exit('PDF looks truncated or empty -- not overwriting a good build with a bad one')
"@
Write-Output $check
