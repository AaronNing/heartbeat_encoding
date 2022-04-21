@echo off

call conda activate ml

cd D:\codes\python\courtship
if exist experiments_cleaned (
  rd /s /q experiments_cleaned
)

mkdir experiments_cleaned
xcopy /s experiments experiments_cleaned

forfiles /p experiments_cleaned /s /m *.ipynb /c "cmd /c jupyter nbconvert --clear-output --inplace @path"

pause