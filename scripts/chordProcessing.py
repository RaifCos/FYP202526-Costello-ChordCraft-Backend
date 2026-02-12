import numpy as np
import librosa

roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def buildChordTemplates():
    rootsLen = len(roots)
    chordTypes = {
        ":maj": [0, 4, 7],           # Major triad
        ":min": [0, 3, 7],           # Minor triad
        ":dim": [0, 3, 6],           # Diminished triad
        ":aug": [0, 4, 8],           # Augmented triad
        ":7": [0, 4, 7, 10],         # Dominant 7th
        ":maj7": [0, 4, 7, 11],      # Major 7th
        ":m7": [0, 3, 7, 10],        # Minor 7th
        ":dim7": [0, 3, 6, 9],       # Diminished 7th
        ":m7b5": [0, 3, 6, 10],      # Half-diminished 7th
        ":sus2": [0, 2, 7],          # Suspended 2nd
        ":sus4": [0, 5, 7],          # Suspended 4th
    }
    
    chordNames = []
    chordTemplates = []
    
    # Generate Template for each Chord.
    for rootIndex in range(rootsLen):
        for chordType, intervals in chordTypes.items():

            # Generate Choord name from Root and Template.
            chordName = f"{roots[rootIndex]}{chordType}"
            chordNames.append(chordName)

            # Binary Row Vector (0 for when note is not in the chord, 1 when it is).
            template = np.zeros(rootsLen, dtype=float)
            for interval in intervals:
                template[(rootIndex + interval) % rootsLen] = 1.0
            chordTemplates.append(template)

    # Create Matrix of Chord Templates
    chordTemplates = np.vstack(chordTemplates).astype(float)

    # Normalize each Template Vector so the algorithm isn't bias towards Chords with more notes. 
    normalizedValues = np.linalg.norm(chordTemplates, axis=1, keepdims=True)
    normalizedValues[normalizedValues == 0] = 1.0 
    chordTemplates /= normalizedValues

    # Return Results.
    return chordNames, chordTemplates

def detectChords(chromagram, hopLength=512, sr=22050):
    frameCount = chromagram.shape[1]
    chordNames, chordTemplates = buildChordTemplates()

    # Compute similarity between each frame and all templates
    # Using cosine similarity (dot product of normalized vectors)
    similarities = np.dot(chordTemplates, chromagram)
    
    # Get best matching chord for each frame
    bestChords = np.argmax(similarities, axis=0)
    
    # Convert frame indices to time
    frameTimes = librosa.frames_to_time(np.arange(frameCount), sr=sr, hop_length=hopLength)
    
    # Group consecutive same chords
    chords = []
    currentChord = bestChords[0]
    startTime = 0
    
    for i in range(1, frameCount):
        if bestChords[i] != currentChord:
            # Chord changed
            endTime = frameTimes[i]
            
            # Only add chords that last at least 0.1 seconds
            if endTime - startTime >= 0.1:
                chords.append({
                    "chord": chordNames[currentChord],
                    "start": float(startTime),
                    "end": float(endTime),
                })            
            currentChord = bestChords[i]
            startTime = endTime
    
    # Add last chord
    chords.append({
        "chord": chordNames[currentChord],
        "start": float(startTime),
        "end": float(frameTimes[-1]),
    })
    return chords
