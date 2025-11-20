import { useBlockchain } from '../context/BlockchainContext'
import Loader from './Loader'

const ConnectWalletButton = () => {
  const { account, connectWallet, disconnectWallet, isLoading } = useBlockchain()

  const handleClick = async () => {
    if (account) {
      disconnectWallet()
    } else {
      await connectWallet()
    }
  }

  if (isLoading) {
    return (
      <div className="glass px-3 py-1.5 rounded-md">
        <span className="text-white text-xs font-medium">Connecting...</span>
      </div>
    )
  }

  return (
    <button
      onClick={handleClick}
      className={`
        relative px-4 py-2 rounded-md font-medium text-xs transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-md
        ${account 
          ? 'bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white shadow-red-500/25' 
          : 'bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600 text-white shadow-green-500/25'
        }
        before:absolute before:inset-0 before:rounded-md before:bg-white/20 before:opacity-0 hover:before:opacity-100 before:transition-opacity
      `}
    >
      <span className="relative z-10">
        {account ? 'Disconnect' : 'Connect Wallet'}
      </span>
    </button>
  )
}

export default ConnectWalletButton
