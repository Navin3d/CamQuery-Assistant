# CamQuery Video Processing Skill

## Skill Overview
This skill defines the video processing capabilities for the CamQuery Assistant.
It explains how to extract frames, analyze them with a vision LLM, store results in vector memory, and answer questions about video content.

## What this skill does
- Extracts frames from uploaded videos at regular intervals
- Uses a vision model to analyze each frame
- Stores frame analysis in a vector database for retrieval
- Provides summaries and timestamps for downstream chat queries

## Inputs
- `video_path` - path to the uploaded video file
- `frame_interval` - optional sampling rate (default: 30)
- `vision_model` - the LLM used for frame-level analysis

## Outputs
- A short status message like `Finished processing <video_name>`
- Saved analysis JSON and summary text in `artifacts/generated/analysis`
- Frame metadata stored in Chroma vector DB

## Tool Functions
### `process_video(video_path: str) -> str`
- Opens the video file
- Saves sampled frames to `artifacts/generated/frames`
- Calls the vision model for frame analysis
- Persists frame metadata and embeddings to the vector database

### `search_video_memory(query: str) -> str`
- Searches the Chroma collection for top matching frames
- Returns consolidated frame summaries for the query

## Usage Examples
- "What happens at 1:20 in the video?"
- "How many people appear in the scene?"
- "Describe the objects visible in the 30th frame."
- "Is it daytime or nighttime in this video?"

## Best Practices
- Process the video before asking questions
- Keep prompts focused on scene details and timestamps
- Use vector search results as the only memory source for accuracy
