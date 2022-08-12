@echo off && setlocal enabledelayedexpansion

CLS
  echo ################################### 
  echo ###  INOVA SISTEMAS ELETRONICOS ###
  echo ################################### 

  echo Usuario: %USERNAME%
  echo Local: %~dp0


testeDisplay.py --conf=Padrao_1.json --ref=%~dp0 --modo=01

pause