
@echo off

SETLOCAL EnableExtensions DisableDelayedExpansion
for /F %%a in ('echo prompt $E ^| cmd') do (
  set "ESC=%%a"
)

SET green=%ESC%[32m
SET reset=%ESC%[0m

SET filename="train_and_test.py"

echo.
echo.
echo %green%Run training...:%reset%
SET args=%*
py -3 src/main.py "src/%filename% %args%"

endlocal