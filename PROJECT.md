# Football Video to Animation MVP

## Product contract
Convert a real football goal clip into a 2D anime/cartoon rendering while preserving source motion, timing and camera. No generative video model.

## Current baseline
- AnimeGANv2 Shinkai ONNX frame stylizer
- Source FPS and dimensions preserved
- Conservative dense optical-flow temporal blending
- High-motion regions prefer the current frame to reduce ghost trails on the ball and players
- Original audio remuxed with FFmpeg when available
- Modern runtime avoids the legacy TensorFlow 1 inference path

## Run
python3 football_cartoon.py input.mp4 -o output.mp4

## Next
Replace Farneback with RAFT, add forward/backward consistency occlusion masks, then football-specific ball/player protection masks and quantitative flicker benchmark.
