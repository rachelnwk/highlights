from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sqlite3
import subprocess
import uuid
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import threading
# it's okay if squiggly, running in virtual env with packages installed

app = Flask(__name__)
CORS(app)  # allows for communication between frontend and backend

UPLOAD_FOLDER = 'uploads'
CLIPS_FOLDER = 'clips'
REELS_FOLDER = 'reels'
MAX_FILE_SIZE = 500 * 1024 * 1024  # = 500MB
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLIPS_FOLDER, exist_ok=True)
os.makedirs(REELS_FOLDER, exist_ok=True)

# Load pretrained YOLO model (i don't want to train my own)
yolo_model = YOLO('yolov8n.pt')  # lightweight model

# Database setup
def init_db():
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY,
            filename TEXT,
            filepath TEXT,
            status TEXT,
            uploaded_at TEXT,
            file_size INTEGER,
            duration REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS highlights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            timestamp REAL,
            duration REAL,
            confidence REAL,
            reason TEXT,
            type TEXT,
            approved INTEGER DEFAULT NULL,
            detected_objects TEXT,
            FOREIGN KEY(video_id) REFERENCES videos(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_duration(filepath):
    """Get video duration using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except:
        return 0

# ============================================
# YOLO-BASED OBJECT DETECTION
# ============================================

def detect_game_events_yolo(filepath, frame_sample_rate=30):
    """
    Use YOLO to detect objects indicating exciting gameplay moments
    
    Detection strategy:
    - High object count = chaotic/action-packed moments
    - Specific objects (weapons, people) = engagement
    - Sudden changes in detections = events happening
    
    For game-specific UI (kill feeds, victory screens), you'd need to:
    1. Train YOLO on your specific game screenshots
    2. Or use OCR for text detection
    """
    highlights = []
    
    try:
        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        detection_scores = []  # Track detection density over time
        frame_timestamps = []
        
        print(f"Analyzing video with YOLO (sampling every {frame_sample_rate} frames)...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified rate
            if frame_count % frame_sample_rate == 0:
                timestamp = frame_count / fps
                
                # Run YOLO detection
                results = yolo_model(frame, verbose=False)
                
                # Count detections and calculate engagement score
                detections = results[0].boxes
                num_objects = len(detections)
                
                # Weight by confidence
                total_confidence = sum([box.conf.item() for box in detections])
                engagement_score = num_objects * (total_confidence / max(num_objects, 1))
                
                detection_scores.append(engagement_score)
                frame_timestamps.append(timestamp)
                
                # Detect specific action-related objects
                detected_classes = [yolo_model.names[int(box.cls.item())] for box in detections]
                
                # High-action indicators
                action_objects = ['person', 'car', 'truck', 'sports ball', 'bottle', 'knife']
                action_count = sum([1 for obj in detected_classes if obj in action_objects])
                
                if action_count > 0 and engagement_score > 2.0:
                    print(f"  Frame at {timestamp:.1f}s: {num_objects} objects detected (score: {engagement_score:.2f})")
            
            frame_count += 1
        
        cap.release()
        
        # Find peaks in detection scores (exciting moments)
        if len(detection_scores) > 0:
            mean_score = np.mean(detection_scores)
            std_score = np.std(detection_scores)
            # Higher threshold to reduce false positives - only very significant peaks
            threshold = mean_score + (1.5 * std_score)  # Much more selective
            
            # Find segments above threshold
            above_threshold = np.array(detection_scores) > threshold
            
            # Group consecutive high-detection frames
            i = 0
            while i < len(above_threshold):
                if above_threshold[i]:
                    start_idx = i
                    # Find end of this peak
                    while i < len(above_threshold) and above_threshold[i]:
                        i += 1
                    end_idx = i
                    
                    # Calculate highlight properties
                    segment_timestamps = frame_timestamps[start_idx:end_idx]
                    segment_scores = detection_scores[start_idx:end_idx]
                    
                    if len(segment_scores) > 0:
                        timestamp = segment_timestamps[0]
                        # Calculate duration based on actual activity length, not fixed
                        activity_duration = segment_timestamps[-1] - segment_timestamps[0]
                        # Smart duration: base on activity + small buffer, more flexible range
                        duration = min(max(activity_duration + 2, 4), 12)  # 4-12 seconds based on activity
                        peak_score = np.max(segment_scores)
                        
                        # More realistic confidence calculation
                        # Base confidence on how much above average the peak is
                        if mean_score > 0:
                            relative_peak = peak_score / mean_score
                            # Sigmoid-like function: 0.5 at 1x mean, 0.8 at 2x mean, 0.95 at 3x mean
                            confidence = min(0.4 + (0.5 / (1 + np.exp(-2 * (relative_peak - 1.5)))), 0.95)
                        else:
                            confidence = 0.3  # Low confidence if no baseline
                        
                        highlights.append({
                            'timestamp': max(0, timestamp - 1),  # Start 1s earlier for context
                            'duration': duration,
                            'confidence': float(confidence),
                            'reason': f'High activity detected (YOLO score: {peak_score:.1f})',
                            'type': 'yolo',
                            'detected_objects': f'{int(peak_score)} objects in scene'
                        })
                else:
                    i += 1
            
            print(f"✓ YOLO analysis complete: Found {len(highlights)} high-activity moments")
        
    except Exception as e:
        print(f"YOLO analysis error: {e}")
    
    return highlights

# ============================================
# AUDIO ANALYSIS
# ============================================

def analyze_audio_volume(filepath):
    """Detect audio spikes (excitement/reactions) using FFmpeg"""
    highlights = []
    
    try:
        # Extract audio volume stats
        cmd = [
            'ffmpeg', '-i', filepath,
            '-af', 'volumedetect',
            '-vn', '-sn', '-dn',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
        
        # Parse mean volume
        mean_volume = -30.0
        for line in result.stdout.split('\n'):
            if 'mean_volume' in line:
                try:
                    mean_volume = float(line.split('mean_volume:')[1].split('dB')[0].strip())
                except:
                    pass
        
        # Detect loud moments (non-silent periods) - more selective
        threshold_db = mean_volume + 12  # 12dB above mean = excitement (more selective)
        
        cmd = [
            'ffmpeg', '-i', filepath,
            '-af', f'silencedetect=noise={threshold_db}dB:d=1',
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
        
        # Parse non-silent periods
        silence_ends = []
        silence_starts = []
        
        for line in result.stdout.split('\n'):
            if 'silence_end' in line:
                try:
                    time = float(line.split('silence_end:')[1].split('|')[0].strip())
                    silence_ends.append(time)
                except:
                    pass
            elif 'silence_start' in line:
                try:
                    time = float(line.split('silence_start:')[1].split()[0])
                    silence_starts.append(time)
                except:
                    pass
        
        # Loud sections are between silence_end and next silence_start
        for i in range(len(silence_ends)):
            start = silence_ends[i]
            end = silence_starts[i] if i < len(silence_starts) else get_video_duration(filepath)
            
            duration = end - start
            if duration > 2:  # At least 2 seconds of excitement
                # More realistic audio confidence based on duration and intensity
                # Longer duration = higher confidence, but cap at reasonable levels
                audio_confidence = min(0.3 + (duration / 10) * 0.4, 0.85)
                
                # Smart audio duration: use actual audio length + small buffer
                audio_duration = min(max(duration + 1, 4), 10)  # 4-10 seconds based on audio length
                
                highlights.append({
                    'timestamp': max(0, start - 0.5),  # Small buffer before audio starts
                    'duration': audio_duration,
                    'confidence': audio_confidence,
                    'reason': 'Audio excitement detected (shouting/reaction)',
                    'type': 'audio',
                    'detected_objects': 'voice activity'
                })
        
        print(f"✓ Audio analysis complete: Found {len(highlights)} audio spikes")
        
    except Exception as e:
        print(f"Audio analysis error: {e}")
    
    return highlights

# ============================================
# COMBINE HIGHLIGHTS
# ============================================

def merge_highlights(yolo_highlights, audio_highlights):
    """Combine YOLO and audio detections, boost confidence when they overlap"""
    all_highlights = []
    
    # Check for overlaps (YOLO + Audio = high confidence hybrid)
    for yh in yolo_highlights:
        overlaps = False
        for ah in audio_highlights:
            # If timestamps within 5 seconds, it's the same event
            if abs(yh['timestamp'] - ah['timestamp']) <= 5:
                overlaps = True
                # Create hybrid highlight with more realistic confidence
                # Hybrid gets a boost but not to 100%
                base_confidence = (yh['confidence'] + ah['confidence']) / 2
                hybrid_boost = 0.15  # 15% boost for having both signals
                hybrid_confidence = min(base_confidence + hybrid_boost, 0.92)  # Cap at 92%
                
                all_highlights.append({
                    'timestamp': min(yh['timestamp'], ah['timestamp']),
                    'duration': max(yh['duration'], ah['duration']),
                    'confidence': hybrid_confidence,
                    'reason': 'Visual action + Audio reaction (HIGH CONFIDENCE)',
                    'type': 'hybrid',
                    'detected_objects': yh['detected_objects']
                })
                break
        
        if not overlaps:
            all_highlights.append(yh)
    
    # Add non-overlapping audio highlights
    for ah in audio_highlights:
        overlaps = False
        for yh in yolo_highlights:
            if abs(ah['timestamp'] - yh['timestamp']) <= 5:
                overlaps = True
                break
        if not overlaps:
            all_highlights.append(ah)
    
    # Sort by confidence, take top 10
    all_highlights.sort(key=lambda x: x['confidence'], reverse=True)
    final = all_highlights[:10]
    
    # Sort by timestamp for display
    final.sort(key=lambda x: x['timestamp'])
    
    print(f"✓ Merged highlights: {len(final)} final highlights")
    return final

def resolve_overlaps(highlights):
    """Resolve overlapping highlights by adjusting durations and filtering too-close highlights"""
    if len(highlights) <= 1:
        return highlights
    
    # Sort by timestamp
    highlights.sort(key=lambda x: x['timestamp'])
    
    resolved = []
    min_gap = 5.0  # Minimum 5 seconds between highlights
    
    for i, current in enumerate(highlights):
        if i == 0:
            resolved.append(current)
            continue
        
        prev = resolved[-1]
        current_start = current['timestamp']
        prev_end = prev['timestamp'] + prev['duration']
        
        # Check if highlights are too close together
        gap = current_start - prev_end
        if gap < min_gap:
            # Skip this highlight if it's too close to the previous one
            print(f"🚫 Skipping highlight at {current_start:.1f}s (too close to previous)")
            continue
        
        # Check for overlap
        if current_start < prev_end:
            overlap = prev_end - current_start
            
            # Adjust the current highlight to start after the previous one
            current['timestamp'] = prev_end + 1.0  # 1 second gap between clips
            
            # Reduce duration if it would make the clip too short
            if current['duration'] > overlap + 2:
                current['duration'] = max(current['duration'] - overlap, 4)
            
            print(f"🔧 Resolved overlap: Adjusted highlight at {current['timestamp']:.1f}s")
        
        resolved.append(current)
    
    print(f"✓ Overlap resolution complete: {len(resolved)} highlights (filtered {len(highlights) - len(resolved)} too-close highlights)")
    return resolved

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if server is running"""
    return jsonify({'status': 'ok', 'yolo_loaded': True})

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use MP4, MOV, AVI, or MKV'}), 400
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return jsonify({'error': 'File too large. Max size is 500MB'}), 400
    
    # Save file
    video_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, f"{video_id}_{filename}")
    file.save(filepath)
    
    duration = get_video_duration(filepath)
    
    # Save to database
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO videos (id, filename, filepath, status, uploaded_at, file_size, duration)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (video_id, filename, filepath, 'uploaded', datetime.now().isoformat(), file_size, duration))
    conn.commit()
    conn.close()
    
    return jsonify({
        'video_id': video_id,
        'filename': filename,
        'status': 'uploaded',
        'file_size_mb': round(file_size / (1024 * 1024), 2),
        'duration': round(duration, 1)
    })

@app.route('/api/analyze/<video_id>', methods=['POST'])
def analyze_video(video_id):
    """Start background analysis"""
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    
    c.execute('SELECT filepath, status FROM videos WHERE id = ?', (video_id,))
    result = c.fetchone()
    
    if not result:
        conn.close()
        return jsonify({'error': 'Video not found'}), 404
    
    filepath, status = result
    
    if status != 'uploaded':
        conn.close()
        return jsonify({'error': 'Video already analyzed or in progress'}), 400
    
    c.execute('UPDATE videos SET status = ? WHERE id = ?', ('analyzing', video_id))
    conn.commit()
    conn.close()
    
    # Run analysis in background thread
    thread = threading.Thread(target=run_analysis, args=(video_id, filepath))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'video_id': video_id,
        'status': 'analyzing',
        'message': 'Analysis started with YOLO + audio detection'
    })

def run_analysis(video_id, filepath):
    """Background analysis task"""
    try:
        print(f"\n=== Starting analysis for {video_id} ===")
        
        # Run YOLO detection
        yolo_highlights = detect_game_events_yolo(filepath, frame_sample_rate=30)
        
        # Run audio analysis
        audio_highlights = analyze_audio_volume(filepath)
        
        # Merge results
        all_highlights = merge_highlights(yolo_highlights, audio_highlights)
        
        # Post-process to handle overlaps
        all_highlights = resolve_overlaps(all_highlights)
        
        # Save to database
        conn = sqlite3.connect('highlights.db')
        c = conn.cursor()
        
        for h in all_highlights:
            c.execute('''
                INSERT INTO highlights (video_id, timestamp, duration, confidence, reason, type, detected_objects)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (video_id, h['timestamp'], h['duration'], h['confidence'], 
                  h['reason'], h['type'], h.get('detected_objects', '')))
        
        c.execute('UPDATE videos SET status = ? WHERE id = ?', ('analyzed', video_id))
        conn.commit()
        conn.close()
        
        print(f"=== Analysis complete for {video_id}: {len(all_highlights)} highlights ===\n")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        conn = sqlite3.connect('highlights.db')
        c = conn.cursor()
        c.execute('UPDATE videos SET status = ? WHERE id = ?', ('error', video_id))
        conn.commit()
        conn.close()

@app.route('/api/videos', methods=['GET'])
def list_videos():
    """Get all videos"""
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    c.execute('SELECT * FROM videos ORDER BY uploaded_at DESC')
    videos = c.fetchall()
    conn.close()
    
    return jsonify([{
        'video_id': v[0],
        'filename': v[1],
        'status': v[3],
        'uploaded_at': v[4],
        'file_size_mb': round(v[5] / (1024 * 1024), 2),
        'duration': v[6]
    } for v in videos])

@app.route('/api/video/<video_id>', methods=['GET'])
def get_video_info(video_id):
    """Get video status and highlights"""
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    
    c.execute('SELECT * FROM videos WHERE id = ?', (video_id,))
    video = c.fetchone()
    
    if not video:
        conn.close()
        return jsonify({'error': 'Video not found'}), 404
    
    c.execute('SELECT * FROM highlights WHERE video_id = ? ORDER BY confidence DESC', (video_id,))
    highlights = c.fetchall()
    conn.close()
    
    return jsonify({
        'video_id': video[0],
        'filename': video[1],
        'status': video[3],
        'uploaded_at': video[4],
        'file_size_mb': round(video[5] / (1024 * 1024), 2),
        'duration': video[6],
        'highlights': [{
            'id': h[0],
            'timestamp': h[2],
            'duration': h[3],
            'confidence': h[4],
            'reason': h[5],
            'type': h[6],
            'approved': h[7],
            'detected_objects': h[8]
        } for h in highlights]
    })

@app.route('/api/highlight/<int:highlight_id>/approve', methods=['POST'])
def approve_highlight(highlight_id):
    """Approve a highlight"""
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    c.execute('UPDATE highlights SET approved = 1 WHERE id = ?', (highlight_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/highlight/<int:highlight_id>/reject', methods=['POST'])
def reject_highlight(highlight_id):
    """Reject a highlight"""
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    c.execute('UPDATE highlights SET approved = 0 WHERE id = ?', (highlight_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/video/<video_id>', methods=['DELETE'])
def delete_video(video_id):
    """Delete a video and all its associated files"""
    print(f"\n=== Deleting video {video_id} ===")
    
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    
    # Get video info
    c.execute('SELECT * FROM videos WHERE id = ?', (video_id,))
    video = c.fetchone()
    
    if not video:
        conn.close()
        print(f"❌ Video {video_id} not found")
        return jsonify({'error': 'Video not found'}), 404
    
    filepath = video[2]
    print(f"📁 Video file: {filepath}")
    
    try:
        # Delete highlights first
        c.execute('DELETE FROM highlights WHERE video_id = ?', (video_id,))
        deleted_highlights = c.rowcount
        print(f"🗑️  Deleted {deleted_highlights} highlights")
        
        # Delete video record
        c.execute('DELETE FROM videos WHERE id = ?', (video_id,))
        deleted_videos = c.rowcount
        print(f"🗑️  Deleted {deleted_videos} video record")
        
        conn.commit()
        conn.close()
        
        # Delete physical files
        files_deleted = []
        
        # Delete main video file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            files_deleted.append(filepath)
            print(f"🗑️  Deleted video file: {filepath}")
        
        # Delete any generated clips
        clips_pattern = os.path.join(CLIPS_FOLDER, f'{video_id}_clip_*.mp4')
        import glob
        for clip_file in glob.glob(clips_pattern):
            if os.path.exists(clip_file):
                os.remove(clip_file)
                files_deleted.append(clip_file)
                print(f"🗑️  Deleted clip: {clip_file}")
        
        # Delete any generated reels
        reel_file = os.path.join(REELS_FOLDER, f'{video_id}_highlights.mp4')
        if os.path.exists(reel_file):
            os.remove(reel_file)
            files_deleted.append(reel_file)
            print(f"🗑️  Deleted reel: {reel_file}")
        
        print(f"✅ Successfully deleted video {video_id}")
        print(f"📁 Files deleted: {len(files_deleted)}")
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'files_deleted': len(files_deleted),
            'highlights_deleted': deleted_highlights
        })
        
    except Exception as e:
        conn.close()
        print(f"❌ Delete error: {e}")
        return jsonify({'error': f'Failed to delete video: {str(e)}'}), 500

@app.route('/api/preview-clip/<video_id>', methods=['POST'])
def generate_preview_clip(video_id):
    """Generate a preview clip for a specific highlight"""
    data = request.get_json()
    timestamp = data.get('timestamp', 0)
    duration = data.get('duration', 5)
    
    print(f"\n=== Generating preview clip for {video_id} ===")
    print(f"Timestamp: {timestamp}s, Duration: {duration}s")
    
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    
    c.execute('SELECT filepath FROM videos WHERE id = ?', (video_id,))
    result = c.fetchone()
    
    if not result:
        conn.close()
        return jsonify({'error': 'Video not found'}), 404
    
    filepath = result[0]
    conn.close()
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Source video file not found'}), 404
    
    try:
        # Generate preview clip
        preview_path = os.path.join(CLIPS_FOLDER, f'{video_id}_preview_{timestamp}_{duration}.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(timestamp),
            '-i', filepath,
            '-t', str(duration),
            '-c:v', 'libx264', '-c:a', 'aac',
            '-preset', 'fast',
            '-vf', 'scale=640:360',  # Resize for faster loading
            preview_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ FFmpeg error: {result.stderr}")
            return jsonify({'error': f'Failed to generate preview: {result.stderr}'}), 500
        
        if not os.path.exists(preview_path):
            return jsonify({'error': 'Preview clip not created'}), 500
        
        print(f"✅ Preview clip created: {preview_path}")
        
        return send_file(preview_path, as_attachment=False, download_name=f'preview_{timestamp}s.mp4')
        
    except Exception as e:
        print(f"❌ Preview generation error: {e}")
        return jsonify({'error': f'Preview generation failed: {str(e)}'}), 500

@app.route('/api/debug/<video_id>', methods=['GET'])
def debug_video(video_id):
    """Debug endpoint to check video and highlight status"""
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    
    # Get video info
    c.execute('SELECT * FROM videos WHERE id = ?', (video_id,))
    video = c.fetchone()
    
    if not video:
        conn.close()
        return jsonify({'error': 'Video not found'}), 404
    
    # Get all highlights
    c.execute('SELECT * FROM highlights WHERE video_id = ?', (video_id,))
    highlights = c.fetchall()
    
    # Count by approval status
    c.execute('SELECT approved, COUNT(*) FROM highlights WHERE video_id = ? GROUP BY approved', (video_id,))
    status_counts = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'video': {
            'id': video[0],
            'filename': video[1],
            'filepath': video[2],
            'status': video[3],
            'file_exists': os.path.exists(video[2]) if video[2] else False
        },
        'highlights': {
            'total': len(highlights),
            'approved': len([h for h in highlights if h[7] == 1]),
            'rejected': len([h for h in highlights if h[7] == 0]),
            'pending': len([h for h in highlights if h[7] is None])
        },
        'status_breakdown': dict(status_counts),
        'highlights_detail': [{
            'id': h[0],
            'timestamp': h[2],
            'duration': h[3],
            'confidence': h[4],
            'reason': h[5],
            'type': h[6],
            'approved': h[7]
        } for h in highlights]
    })

@app.route('/api/generate-reel/<video_id>', methods=['POST'])
def generate_reel(video_id):
    """Generate final highlight reel"""
    print(f"\n=== Generating reel for {video_id} ===")
    
    conn = sqlite3.connect('highlights.db')
    c = conn.cursor()
    
    c.execute('SELECT filepath FROM videos WHERE id = ?', (video_id,))
    result = c.fetchone()
    
    if not result:
        conn.close()
        print(f"❌ Video {video_id} not found")
        return jsonify({'error': 'Video not found'}), 404
    
    filepath = result[0]
    print(f"📁 Source video: {filepath}")
    
    # Check if source file exists
    if not os.path.exists(filepath):
        conn.close()
        print(f"❌ Source file not found: {filepath}")
        return jsonify({'error': 'Source video file not found'}), 404
    
    c.execute('''
        SELECT timestamp, duration FROM highlights 
        WHERE video_id = ? AND approved = 1 
        ORDER BY timestamp
    ''', (video_id,))
    approved = c.fetchall()
    
    # Also check for unapproved highlights
    c.execute('''
        SELECT timestamp, duration FROM highlights 
        WHERE video_id = ? AND approved IS NULL 
        ORDER BY timestamp
    ''', (video_id,))
    unapproved = c.fetchall()
    
    conn.close()
    
    print(f"✅ Approved highlights: {len(approved)}")
    print(f"⚠️  Unapproved highlights: {len(unapproved)}")
    
    # If no approved highlights, use unapproved ones
    if not approved and unapproved:
        print("⚠️  No approved highlights found, using unapproved ones")
        approved = unapproved
    elif not approved:
        print("❌ No highlights found at all")
        return jsonify({'error': 'No highlights found. Please analyze the video first and approve some highlights.'}), 400
    
    try:
        # Extract clips
        clip_files = []
        print(f"🎬 Extracting {len(approved)} clips...")
        
        for i, (timestamp, duration) in enumerate(approved):
            clip_path = os.path.join(CLIPS_FOLDER, f'{video_id}_clip_{i}.mp4')
            print(f"  Clip {i+1}: {timestamp:.1f}s for {duration:.1f}s")
            
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(timestamp),
                '-i', filepath,
                '-t', str(duration),
                '-c:v', 'libx264', '-c:a', 'aac',
                '-preset', 'fast',
                clip_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ FFmpeg error for clip {i}: {result.stderr}")
                continue
                
            if os.path.exists(clip_path):
                clip_files.append(clip_path)
                print(f"  ✅ Created: {clip_path}")
            else:
                print(f"  ❌ Failed to create: {clip_path}")
        
        if not clip_files:
            print("❌ No clips were successfully created")
            return jsonify({'error': 'Failed to create video clips'}), 500
        
        print(f"📝 Concatenating {len(clip_files)} clips...")
        
        # Concatenate clips
        concat_file = os.path.join(CLIPS_FOLDER, f'{video_id}_concat.txt')
        with open(concat_file, 'w') as f:
            for clip in clip_files:
                f.write(f"file '{os.path.abspath(clip)}'\n")
        
        output_path = os.path.join(REELS_FOLDER, f'{video_id}_highlights.mp4')
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ FFmpeg concat error: {result.stderr}")
            return jsonify({'error': f'Failed to concatenate clips: {result.stderr}'}), 500
        
        if not os.path.exists(output_path):
            print(f"❌ Output file not created: {output_path}")
            return jsonify({'error': 'Failed to create final highlight reel'}), 500
        
        print(f"✅ Highlight reel created: {output_path}")
        
        # Cleanup
        try:
            os.remove(concat_file)
            for clip in clip_files:
                if os.path.exists(clip):
                    os.remove(clip)
        except Exception as cleanup_error:
            print(f"⚠️  Cleanup warning: {cleanup_error}")
        
        return send_file(output_path, as_attachment=True, download_name=f'highlights_{video_id}.mp4')
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🎮 Gaming Highlight Reel Generator")
    print("🤖 Powered by YOLOv8 + Audio Analysis")
    print("="*50)
    print("\nBackend running on: http://localhost:5000")
    print("Start the React frontend separately\n")
    app.run(debug=True, host='0.0.0.0', port=5000)