// ============================================
// SETUP INSTRUCTIONS
// ============================================
/*
1. Create React app:
   npx create-react-app highlight-reel-frontend
   cd highlight-reel-frontend

2. Install dependencies:
   npm install axios lucide-react

3. Replace src/App.js with this code

4. Start React app:
   npm start
   (Runs on http://localhost:3000)

5. Make sure backend is running:
   python app.py
   (Runs on http://localhost:5000)
*/

// ============================================
// App.js - Main Application Component
// ============================================

import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Upload, Video, Zap, CheckCircle, XCircle, 
  Download, Play, Trash2, Clock, AlertCircle, Clapperboard
} from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [videos, setVideos] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  // Poll for video updates
  useEffect(() => {
    loadVideos();
    const interval = setInterval(loadVideos, 3000); // Poll every 3 seconds
    return () => clearInterval(interval);
  }, []);

  const loadVideos = async () => {
    try {
      const response = await axios.get(`${API_URL}/videos`);
      setVideos(response.data);
    } catch (error) {
      console.error('Failed to load videos:', error);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });

      setVideos(prev => [response.data, ...prev]);
      e.target.value = '';
    } catch (error) {
      alert('Upload failed: ' + (error.response?.data?.error || error.message));
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const analyzeVideo = async (videoId) => {
    try {
      await axios.post(`${API_URL}/analyze/${videoId}`);
      loadVideos(); // Refresh to show analyzing status
    } catch (error) {
      alert('Analysis failed: ' + (error.response?.data?.error || error.message));
    }
  };

  const deleteVideo = async (videoId) => {
    if (window.confirm('Delete this video? This will permanently remove the video file and all highlights.')) {
      try {
        await axios.delete(`${API_URL}/video/${videoId}`);
        setVideos(prev => prev.filter(v => v.video_id !== videoId));
        console.log('Video deleted successfully');
      } catch (error) {
        console.error('Failed to delete video:', error);
        alert('Failed to delete video: ' + (error.response?.data?.error || error.message));
      }
    }
  };

  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <header className="header">
          <div className="header-content">
            <Clapperboard className="header-icon" size={48} />
            <div>
              <h1>FPS Clip Farmer</h1>
              <p>Detect gaming highlights using AI!</p>
            </div>
          </div>
        </header>

        {/* Upload Section */}
        <div className="upload-section">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          <button
            className="upload-box"
            onClick={() => fileInputRef.current.click()}
            disabled={uploading}
          >
            <Upload size={48} />
            <h3>Upload Gaming Video</h3>
            <p>Max file size: 500MB • MP4, MOV, AVI, MKV</p>
            {uploading && (
              <div className="upload-progress">
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${uploadProgress}%` }} />
                </div>
                <p>{uploadProgress}% uploaded</p>
              </div>
            )}
          </button>
        </div>

        {/* Videos List */}
        <div className="videos-list">
          {videos.length === 0 ? (
            <div className="empty-state">
              <Video size={64} />
              <h3>No videos yet</h3>
              <p>Upload a gaming video to get started</p>
            </div>
          ) : (
            videos.map(video => (
              <VideoCard
                key={video.video_id}
                video={video}
                onAnalyze={analyzeVideo}
                onDelete={deleteVideo}
                onRefresh={loadVideos}
              />
            ))
          )}
        </div>

        {/* Info Footer */}
        <footer className="info-panel">
          <h3>How It Works</h3>
          <div className="info-grid">
            <div className="info-item">
              <div className="info-badge yolo">YOLO</div>
              <h4>Object Detection</h4>
              <p>YOLOv8 detects high-activity moments with multiple objects and movement</p>
            </div>
            <div className="info-item">
              <div className="info-badge audio">Audio</div>
              <h4>Excitement Detection</h4>
              <p>Analyzes volume spikes to detect reactions, shouting, and celebrations</p>
            </div>
            <div className="info-item">
              <div className="info-badge hybrid">Hybrid</div>
              <h4>Combined Intelligence</h4>
              <p>Merges visual and audio signals for high-confidence highlights</p>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

// ============================================
// HighlightCard Component with Video Preview
// ============================================

function HighlightCard({ highlight, videoId, onApprove, onReject }) {
  const [isPlaying, setIsPlaying] = React.useState(false);
  const [videoUrl, setVideoUrl] = React.useState(null);
  const [loading, setLoading] = React.useState(false);

  const generatePreview = async () => {
    if (videoUrl) return; // Already generated
    
    setLoading(true);
    try {
      // Create a preview clip using FFmpeg
      const response = await axios.post(`${API_URL}/preview-clip/${videoId}`, {
        timestamp: highlight.timestamp,
        duration: highlight.duration
      }, { responseType: 'blob' });
      
      const url = URL.createObjectURL(response.data);
      setVideoUrl(url);
    } catch (error) {
      console.error('Failed to generate preview:', error);
      alert('Failed to generate preview clip');
    } finally {
      setLoading(false);
    }
  };

  const handlePlay = () => {
    if (!videoUrl) {
      generatePreview();
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <div className={`highlight-card ${highlight.approved === 1 ? 'approved' : highlight.approved === 0 ? 'rejected' : ''}`}>
      <div className="highlight-preview">
        {videoUrl ? (
          <video 
            src={videoUrl}
            controls
            className="preview-video"
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
          />
        ) : (
          <div className="preview-placeholder" onClick={handlePlay}>
            {loading ? (
              <div className="loading-spinner">Generating preview...</div>
            ) : (
              <div className="play-button">
                <Play size={32} />
                <span>Preview Clip</span>
              </div>
            )}
          </div>
        )}
      </div>
      
      <div className="highlight-info">
        <div className="highlight-meta">
          <span className="timestamp">@{highlight.timestamp.toFixed(1)}s</span>
          <span className="duration">{highlight.duration.toFixed(1)}s</span>
          <span className={`badge badge-${highlight.type}`}>{highlight.type}</span>
        </div>
        
        <div className="highlight-details">
          <p className="reason">{highlight.reason}</p>
          <p className="confidence">Confidence: {Math.round(highlight.confidence * 100)}%</p>
        </div>
        
        <div className="highlight-actions">
          <button 
            className={`btn ${highlight.approved === 1 ? 'btn-success' : 'btn-ghost'}`}
            onClick={() => onApprove(highlight.id)}
          >
            <CheckCircle size={16} />
            Approve
          </button>
          <button 
            className={`btn ${highlight.approved === 0 ? 'btn-danger' : 'btn-ghost'}`}
            onClick={() => onReject(highlight.id)}
          >
            <XCircle size={16} />
            Reject
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================
// VideoCard Component
// ============================================

function VideoCard({ video, onAnalyze, onDelete, onRefresh }) {
  const [details, setDetails] = React.useState(null)
  const [loading, setLoading] = React.useState(false)

  React.useEffect(() => {
    if (video.status === 'analyzed') loadDetails()
  }, [video.status])

  const loadDetails = async () => {
    setLoading(true)
    try {
      const { data } = await axios.get(`${API_URL}/video/${video.video_id}`)
      setDetails(data)
    } catch (e) {
      console.error('Failed to load details:', e)
    } finally {
      setLoading(false)
    }
  }

  const approveHighlight = async (id) => {
    try {
      await axios.post(`${API_URL}/highlight/${id}/approve`)
      await loadDetails()
    } catch (e) {
      console.error(e)
    }
  }

  const rejectHighlight = async (id) => {
    try {
      await axios.post(`${API_URL}/highlight/${id}/reject`)
      await loadDetails()
    } catch (e) {
      console.error(e)
    }
  }

  const generateReel = async () => {
    try {
      const res = await axios.post(`${API_URL}/generate-reel/${video.video_id}`, null, { responseType: 'blob' })
      const url = URL.createObjectURL(res.data)
      const a = document.createElement('a')
      a.href = url
      a.download = `highlights_${video.video_id}.mp4`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      alert('Failed to generate reel')
      console.error(e)
    }
  }

  return (
    <div className="video-card">
      <div className="video-header">
        <div className="video-info">
          <h3>{video.filename}</h3>
          <div className="video-meta">
            <span>Status: {video.status}</span>
            <span>Size: {video.file_size_mb} MB</span>
            {video.duration != null && <span>Duration: {Math.round(video.duration)}s</span>}
          </div>
        </div>
        <div className="video-actions">
          {video.status === 'uploaded' && (
            <button onClick={() => onAnalyze(video.video_id)} className="btn btn-primary">
              Analyze
            </button>
          )}
          {video.status === 'analyzing' && <div className="status status-analyzing">Analyzing…</div>}
          <button onClick={() => onDelete(video.video_id)} className="btn btn-danger">Delete</button>
          <button onClick={onRefresh} className="btn btn-ghost">Refresh</button>
        </div>
      </div>

      {video.status === 'analyzed' && (
        <div className="highlights-section">
          <div>
            <h4>Highlights</h4>
            <button className="btn btn-success" onClick={generateReel}>Download Reel</button>
          </div>

          {loading && <div className="status status-analyzing">Loading details…</div>}

          {details && details.highlights && details.highlights.length > 0 ? (
            <div className="highlights-grid">
              {details.highlights
                .sort((a, b) => a.timestamp - b.timestamp)
                .map(h => (
                  <HighlightCard 
                    key={h.id} 
                    highlight={h} 
                    videoId={video.video_id}
                    onApprove={approveHighlight}
                    onReject={rejectHighlight}
                  />
                ))}
            </div>
          ) : (
            !loading && <div className="status status-analyzing">No highlights yet</div>
          )}
        </div>
      )}
    </div>
  )
}

export default App;

