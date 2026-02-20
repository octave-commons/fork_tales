import { defineConfig } from 'eslint/config'
import baseConfig, { lintBudgets } from './eslint.config.js'

export default defineConfig([
  ...baseConfig,
  {
    files: ['**/*.{ts,tsx}'],
    rules: {
      'sonarjs/cognitive-complexity': ['error', lintBudgets.error.cognitiveComplexity],
      complexity: ['error', { max: lintBudgets.error.cyclomaticComplexity }],
      'max-lines': [
        'error',
        {
          max: lintBudgets.error.maxFileLines,
          skipBlankLines: true,
          skipComments: true,
        },
      ],
    },
  },
])
