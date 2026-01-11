#!/bin/bash

# Niramaya AI - VM Setup Script
# Run this after transferring files to VM

echo "🚀 Setting up Niramaya AI..."

# Load Docker image
echo "📦 Loading Docker image (this may take a few minutes)..."
gunzip -c niramaya-backend.tar.gz | docker load

# Start services
echo "🐳 Starting services..."
docker-compose up -d

# Wait for backend to be ready
echo "⏳ Waiting for backend to start..."
sleep 30

# Check status
echo "✅ Checking service status..."
docker-compose ps

echo ""
echo "🎉 Niramaya AI is now running!"
echo "📍 Access your app at:"
echo "   HTTP:  http://$(curl -s ifconfig.me)"
echo "   HTTPS: https://$(curl -s ifconfig.me)"
echo ""
echo "📊 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"
