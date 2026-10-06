@echo off
cd /d "%~dp0"
"%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe" /nologo /target:winexe /platform:x64 /optimize+ /reference:System.Windows.Forms.dll /reference:System.Drawing.dll /reference:System.IO.Compression.dll /reference:System.IO.Compression.FileSystem.dll /out:"War of the Monsters Launcher.exe" Launcher.cs GameImage.cs GameBuild.cs ModManager.cs LauncherTheme.cs MenuArtwork.cs
exit /b %errorlevel%
