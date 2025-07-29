# PowerShell script for DoRA training on vitraux dataset with DSPO evaluation

# Set paths
$DATASET_PATH = "datasets/vitraux"
$REWARD_MODEL = "outputs/full_enhanced_final/best/model.pt"
$OUTPUT_DIR = "outputs/dora_vitraux_training"

# Training parameters
$BATCH_SIZE = 2
$NUM_STEPS = 2000
$RANK = 16
$LEARNING_RATE = "5e-5"
$EVAL_EVERY = 500
$SAVE_EVERY = 500

Write-Host "🚀 DoRA Training on Vitraux Dataset with DSPO Evaluation" -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "📁 Dataset: $DATASET_PATH" -ForegroundColor Cyan
Write-Host "🏆 Reward Model: $REWARD_MODEL" -ForegroundColor Cyan
Write-Host "💾 Output: $OUTPUT_DIR" -ForegroundColor Cyan
Write-Host "📊 Parameters: batch_size=$BATCH_SIZE, steps=$NUM_STEPS, rank=$RANK" -ForegroundColor Cyan
Write-Host "🔍 DSPO Evaluation: every $EVAL_EVERY steps" -ForegroundColor Cyan
Write-Host ""

# Run training
python scripts/train_dora_vitraux.py `
    --dataset-path "$DATASET_PATH" `
    --reward-model "$REWARD_MODEL" `
    --output-dir "$OUTPUT_DIR" `
    --batch-size $BATCH_SIZE `
    --num-steps $NUM_STEPS `
    --rank $RANK `
    --lr $LEARNING_RATE `
    --eval-every $EVAL_EVERY `
    --save-every $SAVE_EVERY

Write-Host ""
Write-Host "✅ Training completed!" -ForegroundColor Green
Write-Host "📊 Check DSPO results: $OUTPUT_DIR/dspo_evaluation_results.csv" -ForegroundColor Yellow
Write-Host "🏆 Best model: $OUTPUT_DIR/best_model" -ForegroundColor Yellow
Write-Host "📁 Final model: $OUTPUT_DIR/final_model" -ForegroundColor Yellow
