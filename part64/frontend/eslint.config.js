import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import sonarjs from 'eslint-plugin-sonarjs'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export const lintBudgets = {
  warning: {
    cognitiveComplexity: 45,
    cyclomaticComplexity: 40,
    maxFileLines: 1800,
  },
  error: {
    cognitiveComplexity: 180,
    cyclomaticComplexity: 200,
    maxFileLines: 7000,
  },
}

const baseConfig = defineConfig([
  globalIgnores(['dist', 'dist-ssr', 'coverage']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    plugins: {
      sonarjs,
    },
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      'react-hooks/set-state-in-effect': 'off',
      'sonarjs/cognitive-complexity': ['warn', lintBudgets.warning.cognitiveComplexity],
      complexity: ['warn', { max: lintBudgets.warning.cyclomaticComplexity }],
      'max-lines': [
        'warn',
        {
          max: lintBudgets.warning.maxFileLines,
          skipBlankLines: true,
          skipComments: true,
        },
      ],
    },
  },
  {
    files: ['src/components/Simulation/Canvas.tsx'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      'react-hooks/exhaustive-deps': 'off',
      'react-refresh/only-export-components': 'off',
    },
  },
])

export default baseConfig
