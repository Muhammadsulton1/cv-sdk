import asyncio

from src.router import RouterManager

if __name__ == '__main__':
    async def main():
        async with RouterManager() as router:
            await router.process()

    asyncio.run(main())
