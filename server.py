import asyncio
from RunBot import evaluateInput


async def handle_echo(reader, writer):
    data = await reader.read(100)
    message = data.decode()
    botAnswer = evaluateInput(message)
    writer.write(botAnswer.encode())
    await writer.drain()
    writer.close()

loop = asyncio.get_event_loop()
coro = asyncio.start_server(handle_echo, '127.0.0.1', 8888, loop=loop)
server = loop.run_until_complete(coro)

# Serve requests until Ctrl+C is pressed
try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

# Close the server
server.close()
loop.run_until_complete(server.wait_closed())
loop.close()