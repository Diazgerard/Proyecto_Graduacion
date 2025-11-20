const Loader = ({ size = 'md', color = 'blue', text = null }) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  }

  const colorClasses = {
    blue: 'border-blue-500 border-t-blue-200',
    white: 'border-white border-t-white/30',
    green: 'border-green-500 border-t-green-200',
    red: 'border-red-500 border-t-red-200',
    purple: 'border-purple-500 border-t-purple-200'
  }

  const SpinnerComponent = () => (
    <div className="relative">
      <div
        className={`
          ${sizeClasses[size]} 
          border-2 
          ${colorClasses[color]} 
          rounded-full 
          animate-spin
        `}
      ></div>
      {/* Pulse effect */}
      <div
        className={`
          absolute inset-0
          ${sizeClasses[size]} 
          border-2 
          border-transparent
          rounded-full 
          animate-pulse
          ${color === 'white' ? 'bg-white/10' : 'bg-current/10'}
        `}
      ></div>
    </div>
  )

  if (text) {
    return (
      <div className="flex flex-col items-center justify-center space-y-3">
        <SpinnerComponent />
        <p className={`text-sm font-medium ${
          color === 'white' ? 'text-white' : 
          color === 'blue' ? 'text-blue-600' :
          color === 'green' ? 'text-green-600' :
          color === 'red' ? 'text-red-600' :
          'text-purple-600'
        }`}>
          {text}
        </p>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center">
      <SpinnerComponent />
    </div>
  )
}

export default Loader
