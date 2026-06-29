#!/usr/bin/env bash
# Kill whatever is running on port 8000

PID=$(lsof -ti :8000)

if [[ -z "$PID" ]]; then
    echo "포트 8000에 실행 중인 프로세스가 없습니다."
else
    echo "종료 중: PID $PID (port 8000)"
    kill -9 $PID
    echo "완료 ✓"
fi
