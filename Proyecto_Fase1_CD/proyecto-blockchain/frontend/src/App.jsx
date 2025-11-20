import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import Transaction from './pages/Transaction'
import './App.css'

function App() {
  return (
    <div className="min-h-screen">
      <Navbar />
      <div className="container mx-auto px-6 py-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/transactions" element={<Transaction />} />
        </Routes>
      </div>
    </div>
  )
}

export default App
