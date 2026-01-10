#!/bin/bash
# Azure VM Setup Script for Dayflow CDSS
# Run this script on your Azure VM after SSH-ing in

set -e

echo "🚀 Setting up Dayflow CDSS on Azure VM..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install essential packages
echo "📦 Installing essential packages..."
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    htop \
    unzip \
    software-properties-common

# ==================== DOCKER ====================
echo "🐳 Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add current user to docker group
sudo usermod -aG docker $USER

# ==================== NVIDIA DRIVERS & CUDA ====================
echo "🎮 Installing NVIDIA drivers and CUDA..."

# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

# Install NVIDIA driver (if GPU available)
if lspci | grep -i nvidia > /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing drivers..."
    sudo apt-get install -y nvidia-driver-535
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    echo "✅ NVIDIA drivers installed"
else
    echo "⚠️ No NVIDIA GPU detected, skipping driver installation"
fi

# ==================== APPLICATION SETUP ====================
echo "📁 Setting up application directory..."
sudo mkdir -p /opt/dayflow
sudo chown $USER:$USER /opt/dayflow
cd /opt/dayflow

# Create directory structure
mkdir -p backend frontend nginx/ssl

# ==================== FIREWALL ====================
echo "🔥 Configuring firewall..."
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API (internal, optional)
sudo ufw --force enable

# ==================== SWAP (for low memory VMs) ====================
echo "💾 Configuring swap..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# ==================== SYSTEMD SERVICE ====================
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/dayflow.service > /dev/null <<EOF
[Unit]
Description=Dayflow CDSS Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/dayflow
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable dayflow.service

# ==================== FINAL STEPS ====================
echo ""
echo "=============================================="
echo "✅ Azure VM Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Copy your application files to /opt/dayflow/"
echo "2. Copy your ML models (*.pt, *.h5) to /opt/dayflow/backend/"
echo "3. Set up SSL certificates in /opt/dayflow/nginx/ssl/"
echo "4. Configure GitHub secrets (see README)"
echo "5. Push to GitHub to trigger deployment"
echo ""
echo "Manual start: cd /opt/dayflow && docker compose up -d"
echo "View logs: docker compose logs -f"
echo ""
echo "⚠️ REBOOT REQUIRED for NVIDIA drivers!"
echo "Run: sudo reboot"
