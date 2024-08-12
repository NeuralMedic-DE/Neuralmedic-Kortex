import json
import random
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer

class EEGConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        while True:
            data = [random.uniform(-1, 1) for _ in range(8)]  # Simulated EEG data for 8 channels
            await self.send(text_data=json.dumps({
                'values': data
            }))
            await asyncio.sleep(0.0001)  # 10 kHz frequency

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        pass
