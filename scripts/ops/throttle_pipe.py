"""Copy stdin to stdout at no more than RATE bytes/s (token bucket)."""
import sys, time
rate = float(sys.argv[1]); chunk = 1 << 20; t0 = time.monotonic(); sent = 0
src, dst = sys.stdin.buffer, sys.stdout.buffer
while True:
    data = src.read(chunk)
    if not data: break
    dst.write(data); sent += len(data)
    ahead = sent / rate - (time.monotonic() - t0)
    if ahead > 0: time.sleep(ahead)
dst.flush()
