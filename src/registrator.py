import asyncio

from abs_src.abs_registrator import EventRegistrator


class ExampleEventRegistrator(EventRegistrator):

    def process(self) -> None:
        print(self.inference_data)


async def main():
    registrator = ExampleEventRegistrator()
    await registrator.start_process()


if __name__ == '__main__':
    asyncio.run(main())