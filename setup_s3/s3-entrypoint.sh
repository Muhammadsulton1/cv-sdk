#!/bin/sh
set -e

# Функция для получения переменной с fallback на старое имя
get_env_with_fallback() {
    new_key=$1
    old_key=$2
    new_value=$(eval echo \${$new_key:-})
    old_value=$(eval echo \${$old_key:-})

    # Проверяем конфликт: установлены обе переменные
    if [ -n "$new_value" ] && [ -n "$old_value" ]; then
        echo "❌ Конфликт: установлены обе переменные $new_key и $old_key. Используйте только $new_key." >&2
        exit 1
    fi

    # Возвращаем новое значение, если установлено
    if [ -n "$new_value" ]; then
        echo "$new_value"
        return 0
    fi

    # Возвращаем старое значение с предупреждением
    if [ -n "$old_value" ]; then
        echo "⚠️  Переменная $old_key устарела. Используйте $new_key вместо неё." >&2
        echo "$old_value"
        return 0
    fi

    # Ничего не установлено
    return 1
}

# Обратная совместимость: S3_* с fallback на MINIO_*
S3_ACCESS_KEY=$(get_env_with_fallback "S3_ACCESS_KEY" "MINIO_USER") || { echo "❌ S3_ACCESS_KEY (или MINIO_USER) is required"; exit 1; }
S3_SECRET_KEY=$(get_env_with_fallback "S3_SECRET_KEY" "MINIO_PASS") || { echo "❌ S3_SECRET_KEY (или MINIO_PASS) is required"; exit 1; }
export S3_ACCESS_KEY S3_SECRET_KEY

: "${S3_IDENTITY:?S3_IDENTITY is required}"

mkdir -p /etc/seaweedfs

cat > /etc/seaweedfs/s3.json <<EOF
{
  "identities": [
    {
      "name": "${S3_IDENTITY}",
      "credentials": [
        {
          "accessKey": "${S3_ACCESS_KEY}",
          "secretKey": "${S3_SECRET_KEY}"
        }
      ],
      "actions": [
        "Admin",
        "Read",
        "Write",
        "List",
        "Tagging"
      ]
    }
  ]
}
EOF

exec weed "$@"
