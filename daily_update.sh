#!/bin/zsh

# Daily Market Data Update Script
# This script runs the market data cache pipeline for the previous trading day

set -e  # Exit on any error

# Set PATH for launchd compatibility - include Homebrew paths
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${0:A}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/daily_update_$TIMESTAMP.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to send notification (optional - you can customize this)
send_notification() {
    local status=$1
    local message=$2
    
    # macOS notification
    osascript -e "display notification \"$message\" with title \"Market Data Update\" subtitle \"$status\""
    
    # You could also add email notifications, Slack webhooks, etc.
}

# Function to get from date (yesterday, skip weekends for stock data)
get_from_date() {
    local config=$1
    
    if [[ "$config" == "stock" ]]; then
        # For stock data, skip weekends
        local yesterday=$(date -v-1d +"%Y-%m-%d")
        local day_of_week=$(date -jf "%Y-%m-%d" "$yesterday" +"%u")
        
        # If yesterday was Saturday (6) or Sunday (7), go back to Friday
        if [[ "$day_of_week" -eq 6 ]]; then
            yesterday=$(date -v-2d +"%Y-%m-%d")
        elif [[ "$day_of_week" -eq 7 ]]; then
            yesterday=$(date -v-3d +"%Y-%m-%d")
        fi
        
        echo "$yesterday"
    else
        # For forex/crypto, use yesterday (markets run 24/7)
        date -v-1d +"%Y-%m-%d"
    fi
}

# Function to get to date (today)
get_to_date() {
    local config=$1
    
    # For all configs, use today's date
    date +"%Y-%m-%d"
}

# Main execution
main() {
    log_message "Starting daily market data update..."
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Log environment info for debugging
    log_message "Environment info: PPID=$PPID, XPC_SERVICE_NAME=$XPC_SERVICE_NAME, LAUNCHD_SOCKET=$LAUNCHD_SOCKET"
    
    # Source shell configuration to get environment variables
    if [[ -f "$HOME/.zshrc" ]]; then
        source "$HOME/.zshrc"
        log_message "Sourced ~/.zshrc for environment variables"
    fi
    
    # Activate virtual environment
    if [[ -n "$MARKET_VENV_BASE_DIR" && -f "$MARKET_VENV_BASE_DIR/bin/activate" ]]; then
        source "$MARKET_VENV_BASE_DIR/bin/activate"
        log_message "Activated virtual environment from $MARKET_VENV_BASE_DIR"
    elif [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        log_message "Activated local virtual environment"
    else
        log_message "No virtual environment found (checked MARKET_VENV_BASE_DIR and local venv)"
    fi
    
    # Array of configurations to update
    #configs=("crypto" "stock" "forex")
    configs=("crypto")
    
    local overall_success=true
    
    for config in "${configs[@]}"; do
        log_message "Processing $config configuration..."
        
        # Get appropriate dates for this config
        from_date=$(get_from_date "$config")
        to_date=$(get_to_date "$config")
        
        log_message "Updating $config data from $from_date to $to_date"
        
        # Run the cache script with automatic yes and skip check
        if timeout 3600 ./main_cache_common.sh \
            --config "$config" \
            --from "$from_date" \
            --to "$to_date" \
            --skip-check \
            --datatypes "raw,feature,target,resample,feature_resample,ml_data" \
            <<< "y"; then
            
            log_message "✅ Successfully updated $config data"
        else
            log_message "❌ Failed to update $config data"
            overall_success=false
        fi
    done
    
    # Send completion notification
    if [[ "$overall_success" == true ]]; then
        log_message "🎉 Daily market data update completed successfully!"
        send_notification "Success" "All market data updated successfully"
    else
        log_message "⚠️  Daily market data update completed with some failures"
        send_notification "Warning" "Some market data updates failed - check logs"
    fi
    
    # Cleanup old logs (keep last 30 days)
    find "$LOG_DIR" -name "daily_update_*.log" -mtime +30 -delete
    
    log_message "Daily update process finished"
}

# Run main function and capture any errors
if ! main; then
    log_message "❌ Daily update failed with critical error"
    send_notification "Error" "Daily update failed - check logs immediately"
    exit 1
fi 