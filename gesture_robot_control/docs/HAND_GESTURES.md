# Hand Gesture Library

This document describes the hand gestures recognized by our system using MediaPipe Hands. The system supports both static poses and dynamic gestures.

## Static Gestures

### Point 👆
- Index finger extended
- All other fingers closed
- Used for: Basic pointing and as a base for dynamic gestures
- Detection: Index finger distance from palm > 120 pixels, other fingers < 120 pixels

### Fist ✊
- All fingers closed
- Used for: Neutral position / reset
- Detection: All finger distances from palm < 120 pixels

### Open Palm 🖐
- All fingers extended
- Used for: Base position for palm wave
- Detection: All finger distances from palm > 120 pixels

### Hang Loose 🤙
- Thumb and pinky extended
- Other fingers closed
- Used for: Alternative command trigger
- Detection: Thumb and pinky distances from palm > 120 pixels

## Dynamic Gestures

Dynamic gestures are detected by tracking motion patterns over time. The system maintains a history of 20 frames for gesture detection.

### Point + Swipe ↔️
- Base Gesture: Point
- Motion: Horizontal movement while pointing
- Direction: Left or right
- Detection: 
  - Horizontal motion > 100 pixels
  - Vertical motion < 120 pixels
  - Total motion distance > 150 pixels

### Point + Circle ⭕
- Base Gesture: Point
- Motion: Circular movement with index finger
- Detection:
  - Points form approximate circle
  - Mean distance from average radius < 40 pixels
  - Standard deviation of distances < 25 pixels

### Open Palm + Wave 👋
- Base Gesture: Open Palm
- Motion: Fingers oscillating while palm remains stable
- Detection:
  - Palm base movement < 30 pixels (standard deviation)
  - Total fingertip motion > 500 pixels
  - At least 6 direction changes (3 wave cycles)

## Implementation Details

### Motion Tracking
- Frame History: 20 frames
- Gesture Cooldown: 1.0 seconds between dynamic gestures
- Coordinate System: (x,y) in pixel space, normalized by frame size

### Thresholds
- Finger Extension: 120 pixels from palm center
- Palm Stability: 30 pixel standard deviation
- Minimum Swipe Distance: 100 pixels horizontal
- Circle Tolerance: 40 pixel mean deviation from perfect circle

## Usage Notes

1. **Lighting**: Good lighting conditions improve hand landmark detection

2. **Camera Position**: Position camera to capture full hand motion range

3. **Speed**: 
   - Swipes: Moderate, natural motion (more forgiving than before)
   - Circles: Casual circular motion (doesn't need to be perfect)
   - Waves: Natural oscillating motion

4. **Stability**:
   - Keep unused fingers closed for pointing gestures
   - Keep wrist stable for palm wave detection
   - Maintain consistent distance from camera

## Future Enhancements

- [ ] Add rotation detection for point gesture
- [ ] Implement pinch gesture detection
- [ ] Add multi-hand gesture support
- [ ] Improve gesture transition handling
- [ ] Add gesture combination system 