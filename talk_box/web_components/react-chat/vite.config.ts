import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import dts from 'vite-plugin-dts';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const isLib = mode === 'lib';

  return {
    plugins: [
      react(),
      ...(isLib ? [dts({ insertTypesEntry: true })] : [])
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    ...(isLib ? {
      build: {
        lib: {
          entry: path.resolve(__dirname, 'src/index.ts'),
          name: 'TalkBoxChat',
          fileName: 'talk-box-chat',
          formats: ['es', 'umd']
        },
        rollupOptions: {
          external: ['react', 'react-dom'],
          output: {
            globals: {
              react: 'React',
              'react-dom': 'ReactDOM',
            },
          },
        },
        sourcemap: true,
        emptyOutDir: true,
      }
    } : {
      build: {
        outDir: 'build-dist'
      }
    })
  };
});
