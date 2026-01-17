param(
  [string]$Message = "Update"
)

Set-Location C:\docintel

git add -A
git status
git commit -m $Message
git push