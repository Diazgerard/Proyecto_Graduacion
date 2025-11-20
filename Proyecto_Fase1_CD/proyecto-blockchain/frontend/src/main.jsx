import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.jsx'
import { BlockchainProvider } from './context/BlockchainContext'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <BlockchainProvider>
        <App />
      </BlockchainProvider>
    </BrowserRouter>
  </StrictMode>,
)
