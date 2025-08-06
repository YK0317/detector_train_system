"""
Web Frontend Templates and Static Files Manager

Provides basic HTML templates for testing the enhanced API functionality.
"""

import os
from pathlib import Path


class WebTemplateManager:
    """Manages web templates for the training interface"""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
    
    def create_basic_templates(self):
        """Create basic HTML templates for testing"""
        
        # Main dashboard template
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Train System Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .job-card { border: 1px solid #ccc; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .status-running { background-color: #e3f2fd; }
        .status-completed { background-color: #e8f5e8; }
        .status-error { background-color: #ffebee; }
        .progress-bar { width: 100%; height: 20px; background-color: #f0f0f0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background-color: #4caf50; transition: width 0.3s ease; }
        button { padding: 10px 15px; margin: 5px; border: none; border-radius: 3px; cursor: pointer; }
        .btn-primary { background-color: #007bff; color: white; }
        .btn-danger { background-color: #dc3545; color: white; }
        .btn-success { background-color: #28a745; color: white; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 3px; }
        textarea { height: 100px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Train System Dashboard</h1>
        
        <!-- Create New Job Section -->
        <div class="section">
            <h2>Create New Training Job</h2>
            <form id="jobForm">
                <div class="form-group">
                    <label for="jobName">Job Name:</label>
                    <input type="text" id="jobName" name="jobName" required>
                </div>
                
                <div class="form-group">
                    <label for="modelType">Model Type:</label>
                    <select id="modelType" name="modelType">
                        <option value="torchvision">Torchvision</option>
                        <option value="blip">BLIP</option>
                        <option value="generic">Generic</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="epochs">Epochs:</label>
                    <input type="number" id="epochs" name="epochs" value="10" min="1">
                </div>
                
                <div class="form-group">
                    <label for="learningRate">Learning Rate:</label>
                    <input type="number" id="learningRate" name="learningRate" value="0.001" step="0.0001">
                </div>
                
                <div class="form-group">
                    <label for="dataset">Dataset:</label>
                    <input type="file" id="dataset" name="dataset" accept=".zip,.tar,.tar.gz">
                </div>
                
                <button type="submit" class="btn-primary">Create Job</button>
            </form>
        </div>
        
        <!-- Active Jobs Section -->
        <div class="section">
            <h2>Training Jobs</h2>
            <button onclick="loadJobs()" class="btn-primary">Refresh Jobs</button>
            <div id="jobsList"></div>
        </div>
        
        <!-- Real-time Log Section -->
        <div class="section">
            <h2>Real-time Updates</h2>
            <div id="logOutput" style="height: 300px; overflow-y: scroll; background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd;"></div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // API base URL
        const API_BASE = '/api/v1';
        
        // Job form submission
        document.getElementById('jobForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData();
            const config = {
                model: {
                    name: document.getElementById('jobName').value,
                    type: document.getElementById('modelType').value,
                    model_name: 'resnet18',
                    num_classes: 2
                },
                data: {
                    name: 'uploaded_dataset',
                    type: 'image',
                    batch_size: 32
                },
                training: {
                    epochs: parseInt(document.getElementById('epochs').value),
                    learning_rate: parseFloat(document.getElementById('learningRate').value)
                },
                output: {
                    output_dir: '/tmp/training_output',
                    experiment_name: document.getElementById('jobName').value
                }
            };
            
            formData.append('config', JSON.stringify(config));
            
            const datasetFile = document.getElementById('dataset').files[0];
            if (datasetFile) {
                formData.append('dataset', datasetFile);
            }
            
            try {
                const response = await axios.post(`${API_BASE}/jobs`, formData, {
                    headers: {'Content-Type': 'multipart/form-data'}
                });
                
                addLogMessage(`Job created: ${response.data.job_id}`);
                loadJobs();
                document.getElementById('jobForm').reset();
            } catch (error) {
                addLogMessage(`Error creating job: ${error.response.data.error}`);
            }
        });
        
        // Load and display jobs
        async function loadJobs() {
            try {
                const response = await axios.get(`${API_BASE}/jobs`);
                displayJobs(response.data.jobs);
            } catch (error) {
                addLogMessage(`Error loading jobs: ${error.message}`);
            }
        }
        
        // Display jobs in the UI
        function displayJobs(jobs) {
            const jobsList = document.getElementById('jobsList');
            jobsList.innerHTML = '';
            
            jobs.forEach(job => {
                const jobCard = document.createElement('div');
                jobCard.className = `job-card status-${job.status}`;
                
                const progress = job.total_epochs > 0 ? (job.current_epoch / job.total_epochs) * 100 : 0;
                
                jobCard.innerHTML = `
                    <h3>${job.config.model.name || job.id}</h3>
                    <p><strong>Status:</strong> ${job.status}</p>
                    <p><strong>Progress:</strong> ${job.current_epoch}/${job.total_epochs} epochs</p>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <p><strong>Best Accuracy:</strong> ${(job.best_accuracy * 100).toFixed(2)}%</p>
                    <p><strong>Created:</strong> ${new Date(job.created_at).toLocaleString()}</p>
                    ${job.status === 'queued' ? `<button onclick="startJob('${job.id}')" class="btn-success">Start</button>` : ''}
                    ${job.status === 'running' ? `<button onclick="subscribeToJob('${job.id}')" class="btn-primary">Watch Live</button>` : ''}
                    <button onclick="deleteJob('${job.id}')" class="btn-danger">Delete</button>
                `;
                
                jobsList.appendChild(jobCard);
            });
        }
        
        // Start a job
        async function startJob(jobId) {
            try {
                await axios.post(`${API_BASE}/jobs/${jobId}/start`);
                addLogMessage(`Started job: ${jobId}`);
                loadJobs();
            } catch (error) {
                addLogMessage(`Error starting job: ${error.response.data.error}`);
            }
        }
        
        // Delete a job
        async function deleteJob(jobId) {
            if (confirm('Are you sure you want to delete this job?')) {
                try {
                    await axios.delete(`${API_BASE}/jobs/${jobId}`);
                    addLogMessage(`Deleted job: ${jobId}`);
                    loadJobs();
                } catch (error) {
                    addLogMessage(`Error deleting job: ${error.response.data.error}`);
                }
            }
        }
        
        // Subscribe to job updates
        function subscribeToJob(jobId) {
            socket.emit('subscribe_job', {job_id: jobId});
            addLogMessage(`Subscribed to job updates: ${jobId}`);
        }
        
        // Socket event handlers
        socket.on('job_update', (data) => {
            addLogMessage(`Job ${data.job_id} update: Epoch ${data.progress.current_epoch}, Loss: ${data.progress.current_loss}`);
            loadJobs(); // Refresh job display
        });
        
        socket.on('job_completed', (data) => {
            addLogMessage(`Job ${data.job_id} completed successfully!`);
            loadJobs();
        });
        
        socket.on('job_error', (data) => {
            addLogMessage(`Job ${data.job_id} failed: ${data.error}`);
            loadJobs();
        });
        
        // Add log message
        function addLogMessage(message) {
            const logOutput = document.getElementById('logOutput');
            const timestamp = new Date().toLocaleTimeString();
            logOutput.innerHTML += `<div>[${timestamp}] ${message}</div>`;
            logOutput.scrollTop = logOutput.scrollHeight;
        }
        
        // Load jobs on page load
        window.addEventListener('load', () => {
            loadJobs();
            addLogMessage('Dashboard loaded. Ready to create training jobs!');
        });
    </script>
</body>
</html>
        """
        
        dashboard_file = self.template_dir / "dashboard.html"
        with open(dashboard_file, 'w') as f:
            f.write(dashboard_html)
        
        return str(dashboard_file)


def create_web_templates():
    """Create web templates for testing"""
    manager = WebTemplateManager()
    dashboard_path = manager.create_basic_templates()
    return dashboard_path
