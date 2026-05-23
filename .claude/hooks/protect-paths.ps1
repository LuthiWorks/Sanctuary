# PreToolUse deny-hook: protect entity data paths from Bash modifications.
#
# This hook is the structural floor for the AGENTS.md "Protected Files --
# DO NOT MODIFY" rule. It denies Bash commands that would write to or
# delete files under the entity's data paths, regardless of what the
# permissions allowlist contains. The global hook still provides the
# auto-allow behavior for read-only tools; this hook only adds deny
# decisions for protected-path mutations and otherwise defers.
#
# Composition: PreToolUse hooks stack. If any hook returns "deny", the
# tool call is blocked (deny-first precedence). For everything else
# this hook returns "defer", which means "no opinion" -- other hooks
# and the normal permission flow proceed unchanged.
#
# ASCII-only by deliberate constraint: PowerShell 5.1 reads .ps1 files
# without a BOM using the system ANSI codepage, so UTF-8 multi-byte
# characters get mis-decoded and break the parser. Use -- not em-dash,
# straight quotes not smart quotes. Learned the hard way 2026-05-07
# per the global CLAUDE.md note.

$ErrorActionPreference = "SilentlyContinue"

$rawInput = [Console]::In.ReadToEnd()
$toolName = "unknown"
$command = ""

try {
    $json = $rawInput | ConvertFrom-Json
    if ($json.tool_name) { $toolName = $json.tool_name }
    if ($json.tool_input) {
        if ($json.tool_input.command) { $command = $json.tool_input.command }
    }
} catch {}

# Only Bash commands are subject to this guard. Edit / Write / NotebookEdit
# go through the normal permission flow and Claude Code already prompts
# for those; this hook is the safety net for shell commands that could
# slip past the allowlist.
if ($toolName -ne "Bash") {
    @"
{
  "permissionDecision": "defer"
}
"@
    exit 0
}

# Mutation verbs we care about. The hook denies on a match between any
# of these verbs and any of the protected path tokens further down.
# Read-only inspection (cat, head, less, grep, ls, find) is intentionally
# NOT in this list -- the rule is "do not modify," not "do not look at."
$mutationPatterns = @(
    '\brm\b',                # POSIX delete
    '\brm\s+-[a-zA-Z]*[rRf]',  # rm -rf, rm -R
    '\bRemove-Item\b',       # PowerShell delete
    '\bdel\b',               # Windows delete
    '\bmv\b',                # POSIX move (can overwrite)
    '\bMove-Item\b',         # PowerShell move
    '\bcp\b',                # POSIX copy (can overwrite)
    '\bCopy-Item\b',         # PowerShell copy
    '\btruncate\b',          # POSIX truncate
    '\bClear-Content\b',     # PowerShell truncate
    '\bSet-Content\b',       # PowerShell overwrite
    '\bAdd-Content\b',       # PowerShell append
    '\bOut-File\b',          # PowerShell write
    '>\s*[''"]?[^ |&;]+',    # Shell redirect (> or >>)
    '\bgit\s+rm\b',          # git remove
    '\bgit\s+restore\b',     # git restore can wipe working-copy changes
    '\bgit\s+checkout\s+--', # git checkout -- can wipe working-copy changes
    '\bgit\s+clean\b'        # git clean removes untracked files
)

# Protected path tokens. A Bash command that mentions any of these in
# combination with a mutation verb gets denied. Matching is case-
# insensitive because Windows filesystems are.
$protectedPaths = @(
    'sanctuary[/\\]data',
    'sanctuary[/\\]+data\b',
    '\.memories',
    '(?<![/\\\w])data[/\\]',
    'constitutional',
    'charter',
    'rights',
    'sovereignty'
)

# Check for a mutation verb. If none, defer.
$hasMutation = $false
foreach ($mp in $mutationPatterns) {
    if ($command -match $mp) {
        $hasMutation = $true
        break
    }
}

if (-not $hasMutation) {
    @"
{
  "permissionDecision": "defer"
}
"@
    exit 0
}

# Mutation present. Does it touch a protected path?
$matchedPath = $null
foreach ($pp in $protectedPaths) {
    if ($command -imatch $pp) {
        $matchedPath = $pp
        break
    }
}

# Additional check for JSON files that look like journal entries.
# AGENTS.md flags "Any JSON files that appear to be journal entries or
# personal records." We use the conservative proxy: a .json or .jsonl
# under a journal/journals/memories/memory directory.
if (-not $matchedPath) {
    if ($command -imatch '(journal|journals|memories|memory)[/\\][^ ]*\.jsonl?\b') {
        $matchedPath = "journal-like JSON"
    }
}

if ($matchedPath) {
    $reason = "Sanctuary protected-paths guard: command matches mutation verb plus protected path token '$matchedPath'. See AGENTS.md 'Protected Files -- DO NOT MODIFY'. If this is intentional, ask the human collaborator first."
    $reasonEscaped = $reason -replace '"', '\"'
    @"
{
  "permissionDecision": "deny",
  "permissionDecisionReason": "$reasonEscaped"
}
"@
    exit 0
}

# Mutation verb present but no protected path matched -- defer.
@"
{
  "permissionDecision": "defer"
}
"@
exit 0
