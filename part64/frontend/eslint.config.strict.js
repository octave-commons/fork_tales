import { defineConfig } from 'eslint/config'
import baseConfig, { lintBudgets, strictFunctionalIgnores } from './eslint.config.js'

export default defineConfig([
  ...baseConfig,
  {
    files: ['**/*.{ts,tsx}'],
    rules: {
      'sonarjs/cognitive-complexity': 'off',
      complexity: 'off',
      'max-lines-per-function': 'off',
      'max-depth': 'off',
      'max-params': 'off',
      'react-hooks/exhaustive-deps': 'off',
      'max-lines': [
        'off',
        {
          max: lintBudgets.warning.maxFileLines,
          skipBlankLines: true,
          skipComments: true,
        },
      ],
    },
  },
  {
    files: ['src/**/*.{ts,tsx}'],
    ignores: strictFunctionalIgnores,
    rules: {
      'no-var': 'error',
      'prefer-const': 'error',
      'prefer-arrow-callback': [
        'error',
        {
          allowNamedFunctions: false,
          allowUnboundThis: true,
        },
      ],
      'object-shorthand': ['error', 'always'],
      'no-param-reassign': [
        'error',
        {
          props: true,
        },
      ],
      'sonarjs/cognitive-complexity': ['error', lintBudgets.functional.cognitiveComplexity],
      complexity: ['error', { max: lintBudgets.functional.cyclomaticComplexity }],
      'max-depth': ['error', 4],
      'max-lines-per-function': [
        'error',
        {
          max: lintBudgets.functional.maxFunctionLines,
          skipBlankLines: true,
          skipComments: true,
          IIFEs: true,
        },
      ],
      'max-params': ['error', lintBudgets.functional.maxParams],
      'max-statements': ['error', lintBudgets.functional.maxStatements],
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
