#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

VEC_DIR = Path("/Users/seokiho/Desktop/RagInv/data/db_sectional/vecs")
TOTAL   = 886_704
BATCH   = 128
CLEAR   = "\033[2J\033[H"


def fmt_eta(seconds):
    if seconds <= 0:
        return "계산 중..."
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}시간 {m}분"
    elif m > 0:
        return f"{m}분 {s}초"
    return f"{s}초"


def get_latest():
    files = sorted(VEC_DIR.glob("*.npy")) if VEC_DIR.exists() else []
    return int(files[-1].stem) if files else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    log       = []
    prev_stem = get_latest()
    prev_done = prev_stem + BATCH
    prev_time = time.time()
    speed     = 0.0

    try:
        while True:
            time.sleep(args.interval)
            now  = time.time()
            stem = get_latest()
            done = stem + BATCH

            if stem > prev_stem:
                delta = done - prev_done
                dt    = now - prev_time
                if dt > 0 and delta > 0:
                    speed = delta / dt
                log.append((time.strftime("%H:%M:%S"), stem))
                prev_stem = stem
                prev_done = done
                prev_time = now

            pct     = min(done / TOTAL * 100, 100.0)
            bar_len = 40
            filled  = int(bar_len * pct / 100)
            bar     = "█" * filled + "░" * (bar_len - filled)
            eta     = fmt_eta((TOTAL - done) / speed) if speed > 0 else "계산 중..."
            spd     = f"{speed:.1f} 청크/초" if speed > 0 else "계산 중..."

            print(CLEAR, end="", flush=True)
            print(f"{'='*55}")
            print(f"  임베딩 진행 현황 — db_sectional")
            print(f"{'='*55}")
            print(f"  진행: [{bar}] {pct:.2f}%")
            print(f"  완료: {done:,} / {TOTAL:,} 청크")
            print(f"  속도: {spd}")
            print(f"  예상 잔여: {eta}")
            print(f"  갱신: {time.strftime('%H:%M:%S')}")
            print(f"{'─'*55}")
            print(f"  [생성된 vecs 파일 — 총 {len(log)}개]")
            for ts, s in log[-15:]:
                mark = "  ←" if s == log[-1][1] else ""
                print(f"  {ts}  {s:08d}{mark}")
            if not log:
                print("  (대기 중...)")

            if done >= TOTAL:
                print("\n  ✓ 임베딩 완료!")
                break

    except KeyboardInterrupt:
        print("\n모니터 종료.")


if __name__ == "__main__":
    main()