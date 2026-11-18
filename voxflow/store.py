"""Session memory + turn log. In-memory by default; Redis / PostgreSQL when configured."""
import json
import time


class Store:
    def __init__(self, redis_url="", database_url=""):
        self.mem, self.turns, self.redis, self.pg_url, self._pg = {}, [], None, database_url, None
        if redis_url:
            try:
                import redis.asyncio as aioredis
                self.redis = aioredis.from_url(redis_url)
            except ImportError:
                pass

    async def history(self, sid):
        if self.redis:
            try:
                return [json.loads(x) for x in await self.redis.lrange(f"vox:{sid}", -12, -1)]
            except Exception:
                pass
        return self.mem.get(sid, [])[-12:]

    async def append(self, sid, role, content):
        msg = {"role": role, "content": content}
        self.mem.setdefault(sid, []).append(msg)
        if self.redis:
            try:
                await self.redis.rpush(f"vox:{sid}", json.dumps(msg))
                await self.redis.expire(f"vox:{sid}", 3600)
            except Exception:
                pass

    async def log_turn(self, m):
        self.turns.append(m)
        if not self.pg_url:
            return
        try:
            if self._pg is None:
                import asyncpg
                self._pg = await asyncpg.create_pool(self.pg_url, min_size=1, max_size=4)
                await self._pg.execute("CREATE TABLE IF NOT EXISTS turns(ts double precision, session text,"
                                       " ttft_ms real, e2e_ms real, fallback boolean)")
            await self._pg.execute("INSERT INTO turns VALUES($1,$2,$3,$4,$5)",
                                   time.time(), m["session"], m["ttft_ms"], m["e2e_ms"], m["fallback"])
        except Exception:
            pass  # persistence must never break a live call
