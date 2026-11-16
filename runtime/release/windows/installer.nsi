; 2ndPressing Declicker: Windows installer (NSIS 3, Modern UI 2).
; Built on macOS by release/windows/build-release.sh with:
;   makensis -DVERSION=x.y.z -DSTAGE=<staging dir> -DOUTFILE=<setup.exe> installer.nsi
;
; Installs (64-bit Windows 10/11):
;   App   C:\Program Files\2ndPressing\2ndPressing.exe  (+ Start menu shortcut)
;   VST3  C:\Program Files\Common Files\VST3\2ndPressing Declicker.vst3\
;   CLAP  C:\Program Files\Common Files\CLAP\2ndPressing Declicker.clap
; and registers an uninstaller in Settings > Apps.

Unicode true
!include "MUI2.nsh"
!include "x64.nsh"
!include "LogicLib.nsh"

!ifndef VERSION
  !error "Pass -DVERSION=x.y.z"
!endif
!ifndef STAGE
  !error "Pass -DSTAGE=<staging directory>"
!endif
!ifndef OUTFILE
  !define OUTFILE "2ndPressing-${VERSION}-windows-x64-setup.exe"
!endif

!define PRODUCT "2ndPressing"
!define PLUGIN "2ndPressing Declicker"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT}"

Name "${PLUGIN} ${VERSION}"
Caption "${PLUGIN} ${VERSION} Setup"
OutFile "${OUTFILE}"
InstallDir "$PROGRAMFILES64\${PRODUCT}"
RequestExecutionLevel admin
SetCompressor /SOLID lzma
BrandingText "secondpressing.com"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${PLUGIN}"
VIAddVersionKey "CompanyName" "${PRODUCT}"
VIAddVersionKey "FileDescription" "${PLUGIN} installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "GPL-3.0-or-later"

; ---- Pages ----
!define MUI_ICON "${STAGE}/2ndPressing.ico"
!define MUI_UNICON "${STAGE}/2ndPressing.ico"
!define MUI_ABORTWARNING
!define MUI_WELCOMEPAGE_TITLE "${PLUGIN} ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "Click and pop removal for vinyl: a compact neural network finds the clicks, the music on either side rebuilds them, and everything else passes through bit for bit.$\r$\n$\r$\nThis installs the 2ndPressing app and the VST3 and CLAP plugins. You can choose which on the next pages."
!define MUI_COMPONENTSPAGE_SMALLDESC
!define MUI_FINISHPAGE_TITLE "Installed"
!define MUI_FINISHPAGE_TEXT "App: open 2ndPressing from the Start menu.$\r$\n$\r$\nPlugins: restart your DAW and rescan plugins if it doesn't find them.$\r$\n$\r$\nFeedback: hello@secondpressing.com"
!define MUI_FINISHPAGE_LINK "secondpressing.com"
!define MUI_FINISHPAGE_LINK_LOCATION "https://secondpressing.com/"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${STAGE}/LICENSE.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_ICONSTOP "${PLUGIN} needs 64-bit Windows 10 or 11."
    Abort
  ${EndIf}
  SetRegView 64
FunctionEnd

Function un.onInit
  SetRegView 64
FunctionEnd

; ---- Components ----
Section "2ndPressing app" SecApp
  SetOutPath "$INSTDIR"
  File "${STAGE}/app/2ndPressing.exe"
  CreateDirectory "$SMPROGRAMS\${PRODUCT}"
  CreateShortcut "$SMPROGRAMS\${PRODUCT}\${PRODUCT}.lnk" "$INSTDIR\2ndPressing.exe" "" "$INSTDIR\2ndPressing.ico"
SectionEnd

Section "VST3 plugin" SecVST3
  SetOutPath "$COMMONFILES64\VST3\${PLUGIN}.vst3\Contents\x86_64-win"
  File "${STAGE}/vst3/${PLUGIN}.vst3/Contents/x86_64-win/${PLUGIN}.vst3"
SectionEnd

Section "CLAP plugin" SecCLAP
  SetOutPath "$COMMONFILES64\CLAP"
  File "${STAGE}/clap/${PLUGIN}.clap"
SectionEnd

; Always: icon, licence, uninstaller and the Settings > Apps entry.
Section "-common"
  SetOutPath "$INSTDIR"
  File "${STAGE}/2ndPressing.ico"
  File "${STAGE}/LICENSE.txt"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayName" "${PLUGIN}"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKLM "${UNINST_KEY}" "Publisher" "${PRODUCT}"
  WriteRegStr HKLM "${UNINST_KEY}" "URLInfoAbout" "https://secondpressing.com/"
  WriteRegStr HKLM "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\2ndPressing.ico"
  WriteRegStr HKLM "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKLM "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegStr HKLM "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /S'
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD HKLM "${UNINST_KEY}" "NoRepair" 1
SectionEnd

LangString DESC_App ${LANG_ENGLISH} "The standalone app: play a turntable through any audio interface and hear it cleaned live."
LangString DESC_VST3 ${LANG_ENGLISH} "VST3 plugin for most DAWs, in Common Files\VST3."
LangString DESC_CLAP ${LANG_ENGLISH} "CLAP plugin for Bitwig, REAPER and other CLAP hosts, in Common Files\CLAP."
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecApp} $(DESC_App)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecVST3} $(DESC_VST3)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCLAP} $(DESC_CLAP)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ---- Uninstall ----
Section "Uninstall"
  Delete "$INSTDIR\2ndPressing.exe"
  Delete "$INSTDIR\2ndPressing.ico"
  Delete "$INSTDIR\LICENSE.txt"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"
  Delete "$SMPROGRAMS\${PRODUCT}\${PRODUCT}.lnk"
  RMDir "$SMPROGRAMS\${PRODUCT}"
  RMDir /r "$COMMONFILES64\VST3\${PLUGIN}.vst3"
  Delete "$COMMONFILES64\CLAP\${PLUGIN}.clap"
  DeleteRegKey HKLM "${UNINST_KEY}"
SectionEnd
