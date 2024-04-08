# user.py
import torchaudio
from fastapi import APIRouter, Depends, HTTPException, Form
from ..user_database import get_user, verify_user_credentials, update_user_password, get_database
from pydantic import BaseModel
import logging
from fastapi import UploadFile, File
from datetime import datetime
from ..models import User
from ..user_auth import get_current_active_user
import re
from fastapi.encoders import jsonable_encoder
from ..end_points.tts_api import TTS_API
from ..end_points.ttm_api import TTM_API
from ..end_points.vc_api import VC_API
import bittensor as bt
from fastapi.responses import FileResponse
import os
from typing import Annotated
import random
from sqlalchemy.orm import Session



router = APIRouter()
tts_api = TTS_API()
ttm_api = TTM_API()
vc_api = VC_API()



# Define a Pydantic model for the request body
class TTSMrequest(BaseModel):
    prompt: str 

@router.post("/change_password", response_model=dict)
async def change_user_password(
    username: str = Form(...),
    current_password: str = Form(...),
    new_password: str = Form(..., min_length=8, max_length=16, regex="^[a-zA-Z0-9!@#$%^&*()_+{}\[\]:;<>,.?/~\\-=|\\\\]+$"),
    confirm_new_password: str = Form(...),
    current_active_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_database)  # Dependency to provide the session
):
    try:
        # Validate that all required fields are provided
        if not username or not current_password or not new_password or not confirm_new_password:
            bt.logging.error("All fields are required.")
            raise HTTPException(status_code=400, detail="All fields are required.")

        # Check if the username exists
        user = get_user(username, db)
        if not user:
            bt.logging.error("User not found.")
            raise HTTPException(status_code=404, detail="User not found")

        # Verify the user's current password
        if not verify_user_credentials(username, current_password, db):
            bt.logging.error("Invalid credentials.")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Check if the new password and confirm new password match
        if new_password != confirm_new_password:
            bt.logging.error("New password and confirm new password do not match.")
            raise HTTPException(status_code=400, detail="New password and confirm new password do not match.")

        # Check if the new password is different from the current password
        if current_password == new_password:
            bt.logging.error("New password must be different from the current password.")
            raise HTTPException(status_code=400, detail="New password must be different from the current password.")

        # Additional validation: Check if the new password meets the specified conditions
        if not re.match("^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$", new_password):
            bt.logging.error("New password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character.")
            raise HTTPException(status_code=400, detail="New password must contain at least one uppercase letter, one lowercase letter, one digit, and one special character.")

        # Update the user's password
        updated_user = update_user_password(username, new_password, db)
        if not updated_user:
            bt.logging.error("Failed to update password.")
            raise HTTPException(status_code=500, detail="Failed to update password.")

        # Return a success message
        bt.logging.success("Password changed successfully")
        return {"message": "Password changed successfully"}

    except HTTPException as e:
        raise e  # Re-raise HTTPException to return specific error response
    except Exception as e:
        # Log the error for debugging
        logging.error(f"Error during password change: {e}")
        # Return a generic error response
        raise HTTPException(status_code=500, detail="Internal Server Error. Check the server logs for more details.")


##########################################################################################################################


@router.post("/tts_service")
async def tts_service(request: TTSMrequest, user: User = Depends(get_current_active_user)):
    user_dict = jsonable_encoder(user)
    print("User details:", user_dict)
    print("Request details:", request)

    # Check if the user has a subscription
    if user.roles:
        role = user.roles[0]
        if user.subscription_end_time and datetime.utcnow() <= user.subscription_end_time and role.tts_enabled == 1:
            print(f'{user.username}, Congratulations! You have access to Text-to-Speech (TTS) service. Enjoy your experience.')
            
            # Get filtered axons
            # filtered_axons = tts_api.get_filtered_axons()
            filtered_axons = tts_api._generate_filtered_axons_list()
            bt.logging.info(f"Filtered axons: {filtered_axons}")

            # emissions = tts_api.metagraph.E
            # bt.logging.debug(f"all the uids and emissions are in this list: {emissions}")

            # Check if there are axons available
            if not filtered_axons:
                bt.logging.error("No axons available for Text-to-Speech.")
                raise HTTPException(status_code=500, detail="No axons available for Text-to-Speech.")

            # Choose a TTS axon randomly
            uid, axon = random.choice(filtered_axons)
            bt.logging.info(f"Chosen axon: {axon}, UID: {uid}")

            # Use the prompt from the request in the query_network function
            bt.logging.info(f"request prompt: {request.prompt}")
            bt.logging.info(f"request axon here: {axon}")
            response = tts_api.query_network(axon, request.prompt)

            # Process the response
            audio_data = tts_api.process_response(axon, response, request.prompt)
            bt.logging.info(f"Audio data: {audio_data}")

            try:
                file_extension = os.path.splitext(audio_data)[1].lower()  # Extract the file extension from the path
                bt.logging.info(f"audio_file_path: {audio_data}")
            except Exception as e:
                print(e)
                bt.logging.error(f"Error processing audio file path or server unavailable for uid: {uid}")
                raise HTTPException(status_code=404, detail=f"Error processing audio file path or server unavailable for uid: {uid}")
            
            if file_extension not in ['.wav', '.mp3']:
                bt.logging.error("Unsupported audio format.")
                raise HTTPException(status_code=500, detail="Unsupported audio format.")

            # Set the appropriate content type based on the file extension
            content_type = "audio/wav" if file_extension == '.wav' else "audio/mpeg"

            # Return the audio file
            return FileResponse(path=audio_data, media_type=content_type, filename=os.path.basename(audio_data), headers={"TTS-Axon-UID": str(uid)})

        else:
            # If the user doesn't have access to TTM service or subscription is expired, raise 403 Forbidden
            bt.logging.error(f"{user.username}! Your subscription has expired or you do not have access to the Text-to-Speech service.")
            raise HTTPException(status_code=401, detail= f"{user.username}! Your subscription has expired or you do not have access to the Text-to-Speech service.")
    else:
        # If the user doesn't have any roles assigned, raise 403 Forbidden
        bt.logging.error(f"{user.username}! You do not have any roles assigned")
        raise HTTPException(status_code=401, detail=f"{user.username}! You do not have any roles assigned")




# Endpoint for ttm_service
@router.post("/ttm_service")
async def ttm_service(request: TTSMrequest, user: User = Depends(get_current_active_user)):
    user_dict = jsonable_encoder(user)
    print("User details:", user_dict)

    #check if the user has subscription or not
    if user.roles:
        role = user.roles[0]
        if user.subscription_end_time and datetime.utcnow() <= user.subscription_end_time and role.ttm_enabled == 1:
            print("Congratulations! You have access to Text-to-Music (TTM) service.")

            # Get filtered axons
            filtered_axons = ttm_api.get_filtered_axons()
            bt.logging.info(f"Filtered axons: {filtered_axons}")

            # Check if there are axons available
            if not filtered_axons:
                bt.logging.error("No axons available for Text-to-Music.")
                raise HTTPException(status_code=404, detail="No axons available for Text-to-Music.")

            # Choose a TTM axon randomly
            uid, axon = random.choice(filtered_axons)
            bt.logging.info(f"Chosen axon: {axon}, UID: {uid}")
            response = ttm_api.query_network(axon, request.prompt)

            # Process the response
            audio_data = ttm_api.process_response(axon, response, request.prompt)

            try:
                file_extension = os.path.splitext(audio_data)[1].lower()
                bt.logging.info(f"audio_file_path: {audio_data}")
            except Exception as e:
                print(e)
                bt.logging.error(f"Error processing audio file path or server unaviable for uid: {uid}")
                raise HTTPException(status_code=404, detail= f"Error processing audio file path or server unavailable for uid: {uid}")
            # Process each audio file path as needed

            if file_extension not in ['.wav', '.mp3']:
                bt.logging.error(f"Unsupported audio format for uid: {uid}")
                raise HTTPException(status_code=405, detail="Unsupported audio format.")

            # Set the appropriate content type based on the file extension
            content_type = "audio/wav" if file_extension == '.wav' else "audio/mpeg"

            # Return the audio file
            return FileResponse(path=audio_data, media_type=content_type, filename=os.path.basename(audio_data), headers={"TTM-Axon-UID": str(uid)})

        else:
            print(f"{user.username}! You do not have any access to Text-to-Music (TTM) service or subscription is expired.")
            raise HTTPException(status_code=401, detail=f"{user.username}! Your subscription have been expired or you does not have any access to Text-to-Music (TTM) service")
    else:
        print(f"{user.username}! You do not have any roles assigned.")
        raise HTTPException(status_code=401, detail=f"{user.username}! Your does not have any roles assigned")


@router.post("/vc_service")
async def vc_service(audio_file: Annotated[UploadFile, File()], prompt: str = Form(...) , user: User = Depends(get_current_active_user)):
    user_dict = jsonable_encoder(user)
    print("User details:", user_dict)

    # Check if prompt is already enclosed in quotes
    if not prompt.startswith('"') or not prompt.endswith('"'):
        prompt = f'"{prompt}"'  # Add quotes to the prompt if not already quoted

    if prompt:
        bt.logging.info(f"Prompt details: {prompt}")

    # Validate prompt
    if not prompt:
        bt.logging.error("Prompt section cannot be empty.")
        raise HTTPException(status_code=400, detail="Prompt section cannot be empty.")

    # Validate audio file
    if not audio_file:
        bt.logging.error("Audio file is required.")
        raise HTTPException(status_code=400, detail="Audio file is required.")

    if user.roles:
        role = user.roles[0]
        if user.subscription_end_time and datetime.utcnow() <= user.subscription_end_time and role.vc_enabled == 1:
            print("Congratulations! You have access to Voice Clone (VC) service.")

            # Get filtered axons (assuming vc_api is already defined elsewhere)
            filtered_axons = vc_api.get_filtered_axons()
            bt.logging.info(f"Filtered axons: {filtered_axons}")

            # Check if there are axons available
            if not filtered_axons:
                bt.logging.error("No axons available for Voice Clone.")
                raise HTTPException(status_code=500, detail="No axons available for Voice Clone.")

            # Read the audio file and return its content
            temp_file_path = f"temp_audio_file{audio_file.filename}"  # Generate a temporary file name
            with open(temp_file_path, 'wb+') as f:
                f.write(await audio_file.read())  # Write the contents to a temporary file
            waveform, sample_rate = torchaudio.load(temp_file_path)  
            input_audio = waveform.tolist()

            # Choose a VC axon randomly
            uid, axon = random.choice(filtered_axons)
            bt.logging.info(f"Chosen axon: {axon}, UID: {uid}")

            try:
                audio_data = await vc_api.generate_voice_clone(prompt, input_audio, sample_rate, api_axon=[axon], input_file=temp_file_path)
                bt.logging.info(f"audio_file_path: {audio_data}")
            except Exception as e:
                logging.error(f"Error generating voice clone: {e}")
                raise HTTPException(status_code=500, detail="Error generating voice clone")

            # Ensure that audio_data is defined even if an exception occurred
            if not audio_data:
                bt.logging.error("Voice clone audio data not generated.")
                raise HTTPException(status_code=404, detail="Voice clone audio data not generated")

            file_extension = os.path.splitext(audio_data)[1].lower()
            bt.logging.info(f"file_extension: {file_extension}")

            # Process each audio file path as needed
            if file_extension not in ['.wav', '.mp3']:
                bt.logging.error(f"Unsupported audio format for uid: {uid}")
                raise HTTPException(status_code=500, detail="Unsupported audio format.")

            # Return the audio file along with the UID in the response body
            return FileResponse(path=audio_data, media_type="audio/wav", filename=os.path.basename(audio_data), headers={"VC-Axon-UID": str(uid)})

        else:
            print(f"{user.username}! You do not have access to Voice Clone service or subscription is expired.")
            raise HTTPException(status_code=401, detail=f"{user.username}! Your subscription has expired or you do not have access to the Voice Clone service.")
    else:
        print(f"{user.username}! You do not have any roles assigned.")
        raise HTTPException(status_code=401, detail=F"{user.username}! User does not have any roles assigned")