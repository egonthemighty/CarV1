@echo off
REM Package CarV1 project for Google Colab upload
echo Creating CarV1 package for Colab...
echo.

REM Create the ZIP excluding unnecessary files
powershell -Command "Compress-Archive -Path env, tests, config, utils, references, requirements.txt, README.md -DestinationPath CarV1.zip -Force"

echo.
echo ✓ Package created: CarV1.zip
echo.
echo File size:
powershell -Command "(Get-Item CarV1.zip).Length / 1MB | ForEach-Object { '{0:N2} MB' -f $_ }"
echo.
echo Next steps:
echo 1. Open https://colab.research.google.com/
echo 2. Upload CarV1_Colab_Training.ipynb
echo 3. Enable GPU: Runtime → Change runtime type → GPU
echo 4. Run the notebook and upload CarV1.zip when prompted
echo.
pause
