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
    cognitiveComplexity: 28,
    cyclomaticComplexity: 20,
    maxFileLines: 1400,
  },
  functional: {
    cognitiveComplexity: 22,
    cyclomaticComplexity: 18,
    maxFunctionLines: 180,
    maxParams: 6,
    maxStatements: 60,
  },
}

export const strictFunctionalIgnores = [
  '**/*.test.ts',
  '**/*.test.tsx',
  'src/App.tsx',
  'src/components/App/CoreControlPanel.tsx',
  'src/components/App/WorldPanelsViewport.tsx',
  'src/components/Panels/Catalog.tsx',
  'src/components/Panels/Chat.tsx',
  'src/components/Panels/DaimoiPresencePanel.tsx',
  'src/components/Panels/PresenceCallDeck.tsx',
  'src/components/Panels/ProjectionLedgerPanel.tsx',
  'src/components/Panels/RuntimeConfigPanel.tsx',
  'src/components/Panels/StabilityObservatoryPanel.tsx',
  'src/components/Panels/Vitals.tsx',
  'src/components/Panels/WebGraphWeaverPanel.tsx',
  'src/components/Panels/WorldLogPanel.tsx',
  'src/components/Simulation/Canvas.tsx',
  'src/components/Simulation/GalaxyModelDock.tsx',
  'src/hooks/useAutopilotController.ts',
  'src/hooks/useWorldState.ts',
  'src/types/index.ts',
]

const baseConfig = defineConfig([
  globalIgnores(['dist', 'dist-ssr', 'coverage', 'e2e', 'playwright.config.ts']),
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
      'max-depth': ['warn', 4],
      'max-params': ['warn', 5],
      'max-lines-per-function': [
        'warn',
        {
          max: 150,
          skipBlankLines: true,
          skipComments: true,
          IIFEs: true,
        },
      ],
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
