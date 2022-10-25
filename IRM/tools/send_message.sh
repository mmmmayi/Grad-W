#!/usr/bin/env bash

bot_token=1175946578:AAGPmQkVUVWsTT_L5UAcI-HMBGflDiINGN4
user_chat_id=958227045

function send_message() {
    TITLE=$1
    MESSAGE=$2

    curl -s \
        -X POST https://api.telegram.org/bot${bot_token}/sendMessage \
        -d chat_id=${user_chat_id} \
        -d text="<b>${TITLE}</b>%0A${MESSAGE}" \
        -d parse_mode=HTML > /dev/null
}


# send_message "Hello" "Line 1
# Line 2"
title=$1
message=$2
send_message "$title" "$message"
