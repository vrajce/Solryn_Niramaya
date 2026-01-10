# Dayflow CDSS - Azure VM Deployment Guide

## 🚀 Quick Deployment Guide

### Prerequisites
- Azure VM (Ubuntu 22.04 LTS recommended)
- VM Size: `Standard_NC6s_v3` or similar with GPU (for CUDA)
- Or `Standard_D4s_v3` for CPU-only deployment
- GitHub repository with this code
- Domain name (optional, for SSL)

---

## 📋 Step 1: Create Azure VM

### Using Azure CLI:
```bash
# Login to Azure
az login

# Create resource group
az group create --name dayflow-rg --location eastus

# Create VM with GPU (for CUDA)
az vm create \
  --resource-group dayflow-rg \
  --name dayflow-vm \
  --image Ubuntu2204 \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# Or create CPU-only VM (cheaper)
az vm create \
  --resource-group dayflow-rg \
  --name dayflow-vm \
  --image Ubuntu2204 \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# Open required ports
az vm open-port --resource-group dayflow-rg --name dayflow-vm --port 80 --priority 1001
az vm open-port --resource-group dayflow-rg --name dayflow-vm --port 443 --priority 1002
az vm open-port --resource-group dayflow-rg --name dayflow-vm --port 22 --priority 1003
```

### Get VM IP:
```bash
az vm show -g dayflow-rg -n dayflow-vm --show-details --query publicIps -o tsv
```

---

## 📋 Step 2: Setup VM

### SSH into VM:
```bash
ssh azureuser@<VM_PUBLIC_IP>
```

### Run setup script:
```bash
# Download and run setup script
curl -sSL https://raw.githubusercontent.com/<your-username>/<repo>/main/scripts/setup-azure-vm.sh | bash

# Or manually:
git clone https://github.com/<your-username>/<repo>.git /opt/dayflow
cd /opt/dayflow
chmod +x scripts/setup-azure-vm.sh
./scripts/setup-azure-vm.sh

# Reboot for NVIDIA drivers
sudo reboot
```

---

## 📋 Step 3: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `AZURE_VM_HOST` | VM public IP or domain | `20.10.30.40` |
| `AZURE_VM_USER` | SSH username | `azureuser` |
| `AZURE_VM_SSH_KEY` | Private SSH key | Contents of `~/.ssh/id_rsa` |
| `API_URL` | Backend API URL | `https://yourdomain.com/api` |
| `GITHUB_TOKEN` | Auto-provided | (already available) |

### Generate SSH key for deployment:
```bash
# On your local machine
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/github_deploy

# Copy public key to VM
ssh-copy-id -i ~/.ssh/github_deploy.pub azureuser@<VM_IP>

# Copy private key content to GitHub secret AZURE_VM_SSH_KEY
cat ~/.ssh/github_deploy
```

---

## 📋 Step 4: Upload Model Files

The ML models are too large for Git. Upload them separately:

```bash
# From your local machine
scp *.pt *.h5 azureuser@<VM_IP>:/opt/dayflow/backend/

# Or use Azure Blob Storage:
az storage blob upload-batch -d models -s ./models --account-name <storage-account>
```

---

## 📋 Step 5: SSL Setup (Optional but Recommended)

### Using Let's Encrypt:
```bash
# On VM
sudo apt-get install certbot
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem /opt/dayflow/nginx/ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem /opt/dayflow/nginx/ssl/
```

### For testing without SSL:
Edit `nginx/nginx.conf` and comment out the HTTPS server block, keep only HTTP.

---

## 📋 Step 6: Deploy

### Option A: GitHub Actions (Automatic)
Push to `main` or `master` branch:
```bash
git add .
git commit -m "Deploy to Azure"
git push origin main
```

### Option B: Manual Deployment
```bash
# SSH into VM
ssh azureuser@<VM_IP>
cd /opt/dayflow

# Pull latest code
git pull origin main

# Build and start
docker compose -f docker-compose.prod.yml up -d --build

# Check status
docker compose ps
docker compose logs -f
```

---

## 🔧 Useful Commands

```bash
# View logs
docker compose logs -f backend
docker compose logs -f frontend

# Restart services
docker compose restart

# Stop all
docker compose down

# Clean up
docker system prune -af

# Check GPU
nvidia-smi

# Check health
curl http://localhost:8000/health
curl http://localhost/health
```

---

## 🏗️ Project Structure for Deployment

```
/opt/dayflow/
├── backend/
│   ├── Dockerfile
│   ├── api.py
│   ├── requirements.txt
│   ├── router_model.h5
│   ├── bone_model.pt
│   └── lung_model.pt
├── frontend/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── src/
├── nginx/
│   ├── nginx.conf
│   └── ssl/
│       ├── fullchain.pem
│       └── privkey.pem
├── docker-compose.yml
├── docker-compose.prod.yml
└── .github/
    └── workflows/
        └── deploy.yml
```

---

## 💰 Cost Optimization

| VM Type | Specs | Monthly Cost | Use Case |
|---------|-------|--------------|----------|
| Standard_NC6s_v3 | GPU, 6 vCPU, 112GB RAM | ~$900 | Production with CUDA |
| Standard_D4s_v3 | CPU, 4 vCPU, 16GB RAM | ~$140 | Development/Testing |
| Standard_B2s | CPU, 2 vCPU, 4GB RAM | ~$30 | Minimal testing |

### Tips:
- Use Spot VMs for 60-90% savings (can be interrupted)
- Auto-shutdown for dev VMs
- Use Azure Reserved Instances for 30-50% savings

---

## 🐛 Troubleshooting

### Docker issues:
```bash
# Check Docker status
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Check container logs
docker logs dayflow-backend
```

### GPU not detected:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall NVIDIA container toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Port already in use:
```bash
# Find process using port
sudo lsof -i :80
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>
```

### Out of memory:
```bash
# Add swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```
