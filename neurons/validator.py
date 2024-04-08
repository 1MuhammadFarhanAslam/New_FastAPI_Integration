import os
import sys
import asyncio
import datetime as dt
import wandb
import bittensor as bt
import uvicorn
from pyngrok import ngrok  # Import ngrok from pyngrok

# Set the project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Set the 'AudioSubnet' directory path
audio_subnet_path = os.path.abspath(project_root)

# Add the project root and 'AudioSubnet' directories to sys.path
sys.path.insert(0, project_root)
sys.path.insert(0, audio_subnet_path)

from lib.globals import service_flags
from classes.tts import TextToSpeechService 
from classes.vc import VoiceCloningService
from classes.ttm import MusicGenerationService
from classes.aimodel import AIModelService

# Check if the 'app' folder exists
if os.path.exists(os.path.join(project_root, 'app')):
    from app.fastapi_server import create_app

async def run_fastapi_with_ngrok(app):
    # Setup ngrok tunnel
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)
    try:
        # Run the server using uvicorn
        config = uvicorn.Config(app=app, host="0.0.0.0", port=40337)
        server = uvicorn.Server(config)
        await server.serve()
    finally:
        # Close ngrok tunnel when server exits
        ngrok_tunnel.close()

class AIModelController():
    def __init__(self):
        self.aimodel = AIModelService()
        self.text_to_speech_service = TextToSpeechService()
        self.music_generation_service = MusicGenerationService()
        self.voice_cloning_service = VoiceCloningService()
        self.current_service = self.text_to_speech_service
        self.service = service_flags
        self.last_run_start_time = dt.datetime.now()

    async def run_services(self):
        while True:
            self.check_and_update_wandb_run()
            if isinstance(self.current_service, TextToSpeechService) and self.service["TextToSpeechService"]:
                await self.current_service.run_async()
                self.current_service = self.music_generation_service
            elif isinstance(self.current_service, MusicGenerationService) and self.service["MusicGenerationService"]:
                await self.current_service.run_async()
                self.current_service = self.voice_cloning_service
            elif isinstance(self.current_service, VoiceCloningService) and self.service["VoiceCloningService"]:
                await self.current_service.run_async()
                self.current_service = self.text_to_speech_service

    def check_and_update_wandb_run(self):
        # Calculate the time difference between now and the last run start time
        current_time = dt.datetime.now()
        time_diff = current_time - self.last_run_start_time
        # Check if 4 hours have passed since the last run start time
        if time_diff.total_seconds() >= 4 * 3600:  # 4 hours * 3600 seconds/hour
            self.last_run_start_time = current_time  # Update the last run start time to now
            if self.wandb_run:
                wandb.finish()  # End the current run
            self.new_wandb_run()  # Start a new run

    def new_wandb_run(self):
        now = dt.datetime.now()
        run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
        name = f"Validator-{self.aimodel.uid}-{run_id}"
        commit = self.aimodel.get_git_commit_hash()
        self.wandb_run = wandb.init(
            name=name,
            project="AudioSubnet_Valid",
            entity="subnet16team",
            config={
                "uid": self.aimodel.uid,
                "hotkey": self.aimodel.wallet.hotkey.ss58_address,
                "run_name": run_id,
                "type": "Validator",
                "tao (stake)": self.aimodel.metagraph.neurons[self.aimodel.uid].stake.tao,
                "commit": commit,
            },
            tags=self.aimodel.sys_info,
            allow_val_change=True,
            anonymous="allow",
        )
        bt.logging.debug(f"Started a new wandb run: {name}")

async def main():
    controller = AIModelController()
    controller.new_wandb_run()
    await controller.run_services()

    # Initialize an empty list to hold our tasks
    tasks = []

    # Iterate through each service and create an asynchronous task for its run_async method
    services = [
        controller.text_to_speech_service,
        controller.music_generation_service,
        controller.voice_cloning_service,
    ]
    for service in services:
        if isinstance(service, TextToSpeechService):
            service.new_wandb_run()  # Initialize the Weights & Biases run if the service is TextToSpeechService
        task = asyncio.create_task(service.run_async())
        tasks.append(task)

        await asyncio.sleep(0.1)  # Short delay between task initializations if needed

    # If the 'app' folder exists, create and run the FastAPI app
    if os.path.exists(os.path.join(project_root, 'app')):
        # Read secret key from environment variable
        secret_key = os.getenv("AUTH_SECRET_KEY")
        if not secret_key:
            raise ValueError("Auth Secret key not found in environment variable AUTH_SECRET_KEY")
        app = create_app(secret_key)
        # Create a task for running FastAPI with ngrok
        fastapi_task = asyncio.create_task(run_fastapi_with_ngrok(app))

        # Wait for all tasks to complete, prioritizing the FastAPI task
        await asyncio.gather(fastapi_task, *tasks)
    else:
        # If the 'app' folder does not exist, continue running other tasks normally
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())