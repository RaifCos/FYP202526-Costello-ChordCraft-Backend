# :musical_keyboard: ChordCraft - Backend Code 
ChordCraft is an Android Application built to help users identify chordal features in any audio provided. This project is divided into two main repositories:

- **Frontend Application** ([Link](https://github.com/RaifCos/FYP202526-Costello-ChordCraft-Frontend))
- **Backend API** (Current Repo)

## 	:guitar: Automatic Chord Recognition (ACR) Model
This repository contains an implementation of the chord-CNN-LSTM AI model, which can detect and return chordal information of a given audio file. 

The model is ran using the ```scripts/runModel.py``` handler script, which calls the ```chord_recognition``` module, passing through the path of the target audio file. The model returns a python object to the handler script, which is then seralized into a JSON-formatted string. The output includes each chord detected in the audio processed, with timestamps denoting when in the audio the chord begins and ends. 

## :computer: API Handling
The ```app/main.py``` script handles incoming API requests. When a request is made to the ```/run``` endpoint, the script checks that an audio file has been provided to process, before storing it in a temporary location so it can be accessed by the ACR model. The API script then calls the model handler script, returning the final result to the user. The API script also conatains error handling for invalid file formats and any ACR model errors. 

To ensure end-user privacy, the temporary file will always be removed after processing, regardless if the model succeeded or not. 

## :outbox_tray: CI/CD Pipeline
This Repo implements strong CI/CD practices by using GitHub workflows to package and push each iteration of the API code as a Docker Image. The ```fyp202526-costello-chordcraft-backend``` image is then hosted through Railway Cloud Services, which provides comprehensive logs for Deployment, HTTP Requests, and Network Flow. This allows for efficient monitering of API performance, traffic, and error identification. 

## :round_pushpin: About this Application
The ChordCraft is being developed as part of University of Galway CT413 Final Year Project module FY25/26 with the project title "ChordCraft - Audio-to-Guitar Chords". 
- Student Name: Raif Costello
- SID: 22318961
