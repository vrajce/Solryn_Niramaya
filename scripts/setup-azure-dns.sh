#!/bin/bash
# Script to set up Azure DNS name label for the VM
# Run this locally with Azure CLI installed and logged in

# Configuration - Update these values
RESOURCE_GROUP="Nirmaya-AI_group"  # Your Azure resource group
VM_NAME="Nirmaya-AI"               # Your VM name
DNS_LABEL="nirmaya-ai"             # DNS label (will be: nirmaya-ai.<region>.cloudapp.azure.com)

echo "🔧 Setting up DNS name label for Azure VM..."

# Get the public IP resource name
PUBLIC_IP_NAME=$(az vm show -g $RESOURCE_GROUP -n $VM_NAME --query "networkProfile.networkInterfaces[0].id" -o tsv | xargs -I {} az network nic show --ids {} --query "ipConfigurations[0].publicIPAddress.id" -o tsv | xargs -I {} az network public-ip show --ids {} --query "name" -o tsv)

if [ -z "$PUBLIC_IP_NAME" ]; then
    echo "❌ Could not find public IP for VM"
    exit 1
fi

echo "📍 Found Public IP: $PUBLIC_IP_NAME"

# Set the DNS name label
az network public-ip update \
    --resource-group $RESOURCE_GROUP \
    --name $PUBLIC_IP_NAME \
    --dns-name $DNS_LABEL

# Get the FQDN
FQDN=$(az network public-ip show \
    --resource-group $RESOURCE_GROUP \
    --name $PUBLIC_IP_NAME \
    --query "dnsSettings.fqdn" -o tsv)

echo ""
echo "✅ DNS name configured successfully!"
echo "🌐 Your domain: $FQDN"
echo ""
echo "📋 Next steps:"
echo "1. Add this secret to GitHub: AZURE_DOMAIN=$FQDN"
echo "2. Add this secret to GitHub: SSL_EMAIL=your-email@example.com"
echo "3. Update API_URL secret to: https://$FQDN/api"
echo "4. Push to trigger deployment"
