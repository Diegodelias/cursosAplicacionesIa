-- Script SQL para sistema de inicio de sesión / SQL Script for login system
-- Generado por Ollama a través de LangChain / Generated by Ollama via LangChain

-- Tabla inicial y consultas / Initial table and queries
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inserting a new user
INSERT INTO users (username, password)
VALUES ('johnDoe', 'hashed_password_here');

-- Retrieving all user data
SELECT * FROM users;

-- Retrieving a specific user by their username
SELECT id, username, created_at
FROM users
WHERE username = 'johnDoe';

-- Tabla modificada con email / Modified table with email
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inserting a new user with email address
INSERT INTO users (username, email, password)
VALUES ('johnDoe', 'johndoe@example.com', 'hashed_password_here');

-- Retrieving all user data
SELECT * FROM users;

-- Retrieving a specific user by their username or email
SELECT id, username, email, created_at
FROM users
WHERE 
  (username = 'johnDoe' OR email = 'johndoe@example.com');

-- Checking if a user exists by their email address
SELECT id, username, created_at
FROM users
WHERE email = 'newuser@example.com';