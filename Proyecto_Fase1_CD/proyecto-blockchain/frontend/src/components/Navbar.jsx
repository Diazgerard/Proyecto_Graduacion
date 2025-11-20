import { Link, useLocation } from 'react-router-dom'
import { useBlockchain } from '../context/BlockchainContext'
import ConnectWalletButton from './ConnectWalletButton'

const Navbar = () => {
  const location = useLocation()
  const { account } = useBlockchain()

  const isActive = (path) => {
    return location.pathname === path
  }

  const formatAddress = (address) => {
    if (!address) return ''
    return `${address.slice(0, 6)}...${address.slice(-4)}`
  }

  return (
    <nav className="glass sticky top-0 z-50 backdrop-blur-md">
      <div className="container mx-auto px-4 py-3">
        <div className="flex justify-between items-center">
          {/* Logo/Brand */}
          <Link to="/" className="flex items-center group">
            <span className="text-lg font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
              CryptoHub
            </span>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-1">
            <Link
              to="/"
              className={`relative px-3 py-2 rounded-md text-sm transition-all duration-300 ${
                isActive('/') 
                  ? 'text-white bg-white/20' 
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              <span>Home</span>
              {isActive('/') && (
                <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-pink-400 rounded-md opacity-20"></div>
              )}
            </Link>
            <Link
              to="/dashboard"
              className={`relative px-3 py-2 rounded-md text-sm transition-all duration-300 ${
                isActive('/dashboard') 
                  ? 'text-white bg-white/20' 
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              <span>Dashboard</span>
              {isActive('/dashboard') && (
                <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-pink-400 rounded-md opacity-20"></div>
              )}
            </Link>
            <Link
              to="/transactions"
              className={`relative px-3 py-2 rounded-md text-sm transition-all duration-300 ${
                isActive('/transactions') 
                  ? 'text-white bg-white/20' 
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              <span>Transactions</span>
              {isActive('/transactions') && (
                <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-pink-400 rounded-md opacity-20"></div>
              )}
            </Link>
          </div>

          {/* Wallet Connection Status & Button */}
          <div className="flex items-center space-x-2">
            {account && (
              <div className="hidden md:flex glass px-3 py-1.5 rounded-md">
                <span className="text-white text-xs font-mono">
                  {formatAddress(account)}
                </span>
              </div>
            )}
            <ConnectWalletButton />
          </div>
        </div>

        {/* Mobile Navigation */}
        <div className="md:hidden mt-3 pt-3 border-t border-white/10">
          <div className="flex flex-col space-y-1">
            <Link
              to="/"
              className={`px-3 py-2 rounded-md transition-all duration-300 text-center ${
                isActive('/') 
                  ? 'text-white bg-white/20' 
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              <span className="text-sm">Home</span>
            </Link>
            <Link
              to="/dashboard"
              className={`px-3 py-2 rounded-md transition-all duration-300 text-center ${
                isActive('/dashboard') 
                  ? 'text-white bg-white/20' 
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              <span className="text-sm">Dashboard</span>
            </Link>
            <Link
              to="/transactions"
              className={`px-3 py-2 rounded-md transition-all duration-300 text-center ${
                isActive('/transactions') 
                  ? 'text-white bg-white/20' 
                  : 'text-gray-300 hover:text-white hover:bg-white/10'
              }`}
            >
              <span className="text-sm">Transactions</span>
            </Link>
            {account && (
              <div className="px-3 py-2 mt-1 glass rounded-md text-center">
                <span className="text-white text-sm font-mono">
                  {formatAddress(account)}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
