#!/bin/bash
# Cleanup script to remove redundant files

echo "🧹 Cleaning up redundant files..."

# Remove old Streamlit app versions
rm -f app.py pre_final_app.py pre_pre_final_app.py
rm -f pre_pre_symptoms_solver.py pre_symptoms_solver.py symptom_solver.py

# Remove backup model file
rm -f bone_model_backup_real.pt

# Remove redundant backend folder (we use root-level api.py now)
rm -rf backend/

# Remove Python cache
rm -rf __pycache__

# Remove old frontend build (will be regenerated)
# rm -rf frontend-dist/

echo "✅ Cleanup complete!"
echo ""
echo "Kept files:"
echo "  - api.py (main backend)"
echo "  - bone_model.pt, lung_model.pt, lung-tb-model.pt"
echo "  - router_model.h5, best_model_auc.keras, pneumonia_unet_v2.h5"
echo "  - frontend/ (React app)"
echo "  - docker-compose.yml, Dockerfile"
echo "  - Test folders (for validation)"
