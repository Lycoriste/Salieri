#!/bin/bash

KEY=$(openssl rand -base64 48)

echo "Generated:"
echo "$KEY"
echo "$KEY" > .env
