#!/usr/bin/env node

/**
 * Cleanup script to remove files from outputs and uploads directories
 * Runs before starting the development server
 */

import { rmSync, mkdirSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

const dirsToClean = [
  join(projectRoot, 'outputs'),
  join(projectRoot, 'uploads')
];

console.log('🧹 Cleaning up directories before starting server...');

dirsToClean.forEach(dir => {
  if (existsSync(dir)) {
    try {
      // Get all files and subdirectories
      const items = readdirSync(dir);

      if (items.length === 0) {
        console.log(`   ✓ ${dir.replace(projectRoot, '.')} - already empty`);
        return;
      }

      // Remove all contents
      items.forEach(item => {
        const itemPath = join(dir, item);
        try {
          rmSync(itemPath, { recursive: true, force: true });
        } catch (err) {
          console.warn(`   ⚠ Warning: Could not remove ${item}:`, err.message);
        }
      });

      console.log(`   ✓ ${dir.replace(projectRoot, '.')} - cleaned (${items.length} items removed)`);
    } catch (err) {
      console.error(`   ✗ Error cleaning ${dir}:`, err.message);
    }
  } else {
    // Create directory if it doesn't exist
    try {
      mkdirSync(dir, { recursive: true });
      console.log(`   ✓ ${dir.replace(projectRoot, '.')} - created`);
    } catch (err) {
      console.error(`   ✗ Error creating ${dir}:`, err.message);
    }
  }
});

console.log('✅ Cleanup complete!\n');
