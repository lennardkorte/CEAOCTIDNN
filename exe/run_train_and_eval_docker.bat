
SETLOCAL EnableExtensions DisableDelayedExpansion
for /F %%a in ('echo prompt $E ^| cmd') do (
  set "ESC=%%a"
)

SET green=%ESC%[32m
SET reset=%ESC%[0m

SET name_image="image-iddatdloct"
SET name_container="container-iddatdloct"
SET filename="train_and_test.py"

echo.
echo.
echo %green%Building docker-image...%reset%
docker build -t %name_image% .

echo.
echo.
echo %green%Removing additional <none> images...%reset%
docker rm $(docker ps -a -q) > /dev/null 2>&1
docker image prune -f

echo.
echo.
echo %green%Show all images:%reset%
docker image ls

echo.
echo.
echo %green%Run docker-image:%reset%
SET args=%*
docker run ^
-it --rm ^
--gpus all ^
--shm-size 8G ^
--name %name_container% ^
--mount type=bind,source=/home/Korte/IDDATDLOCT/data,target=/IDDATDLOCT/data ^
-i %name_image% "src/%filename% %args%"

endlocal