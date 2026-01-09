/**
 * Custom JavaScript to add Sign-Up functionality to Chainlit login page
 */

(function() {
    'use strict';

    // Wait for DOM to be fully loaded
    function init() {
        // Find the login form
        const checkForLoginForm = setInterval(() => {
            const loginContainer = document.querySelector('[class*="authLogin"]') ||
                                 document.querySelector('form') ||
                                 document.querySelector('[class*="MuiBox"]');

            if (loginContainer) {
                clearInterval(checkForLoginForm);
                addSignUpButton(loginContainer);
            }
        }, 100);

        // Stop checking after 10 seconds
        setTimeout(() => clearInterval(checkForLoginForm), 10000);
    }

    function addSignUpButton(loginContainer) {
        // Check if button already exists
        if (document.getElementById('custom-signup-btn')) {
            return;
        }

        // Create sign-up button container
        const signupContainer = document.createElement('div');
        signupContainer.style.cssText = `
            margin-top: 16px;
            text-align: center;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.12);
        `;

        // Create sign-up text
        const signupText = document.createElement('p');
        signupText.style.cssText = `
            margin: 0 0 12px 0;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.7);
        `;
        signupText.textContent = "Don't have an account?";

        // Create sign-up button
        const signupButton = document.createElement('button');
        signupButton.id = 'custom-signup-btn';
        signupButton.textContent = 'Create Account';
        signupButton.style.cssText = `
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        `;

        signupButton.onmouseover = function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.5)';
        };

        signupButton.onmouseout = function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 2px 8px rgba(102, 126, 234, 0.3)';
        };

        signupButton.onclick = function(e) {
            e.preventDefault();
            showSignUpModal();
        };

        signupContainer.appendChild(signupText);
        signupContainer.appendChild(signupButton);

        // Try to find the best place to insert
        const submitButton = loginContainer.querySelector('button[type="submit"]');
        if (submitButton && submitButton.parentElement) {
            submitButton.parentElement.appendChild(signupContainer);
        } else {
            loginContainer.appendChild(signupContainer);
        }
    }

    function showSignUpModal() {
        // Create modal overlay
        const modal = document.createElement('div');
        modal.id = 'signup-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            animation: fadeIn 0.3s ease;
        `;

        // Create modal content
        const modalContent = document.createElement('div');
        modalContent.style.cssText = `
            background: #1e1e1e;
            padding: 32px;
            border-radius: 12px;
            max-width: 400px;
            width: 90%;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s ease;
        `;

        modalContent.innerHTML = `
            <style>
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                @keyframes slideIn {
                    from { transform: translateY(-20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
                #signup-form input {
                    width: 100%;
                    padding: 12px;
                    margin: 8px 0;
                    border: 2px solid rgba(255, 255, 255, 0.1);
                    border-radius: 6px;
                    background: rgba(255, 255, 255, 0.05);
                    color: white;
                    font-size: 14px;
                    box-sizing: border-box;
                }
                #signup-form input:focus {
                    outline: none;
                    border-color: #667eea;
                    background: rgba(255, 255, 255, 0.08);
                }
                #signup-form label {
                    display: block;
                    margin-top: 12px;
                    margin-bottom: 4px;
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 14px;
                    font-weight: 500;
                }
                .hint {
                    font-size: 12px;
                    color: rgba(255, 255, 255, 0.5);
                    margin-top: 4px;
                }
                .modal-buttons {
                    display: flex;
                    gap: 12px;
                    margin-top: 24px;
                }
                .modal-buttons button {
                    flex: 1;
                    padding: 12px;
                    border: none;
                    border-radius: 6px;
                    font-size: 14px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                #signup-submit {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                #signup-submit:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
                }
                #signup-submit:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                    transform: none;
                }
                #signup-cancel {
                    background: rgba(255, 255, 255, 0.1);
                    color: white;
                }
                #signup-cancel:hover {
                    background: rgba(255, 255, 255, 0.15);
                }
                .error-message {
                    background: rgba(244, 67, 54, 0.2);
                    border: 1px solid rgba(244, 67, 54, 0.5);
                    color: #ff6b6b;
                    padding: 12px;
                    border-radius: 6px;
                    margin-top: 12px;
                    font-size: 13px;
                }
                .success-message {
                    background: rgba(76, 175, 80, 0.2);
                    border: 1px solid rgba(76, 175, 80, 0.5);
                    color: #66bb6a;
                    padding: 12px;
                    border-radius: 6px;
                    margin-top: 12px;
                    font-size: 13px;
                }
            </style>
            <h2 style="margin: 0 0 8px 0; color: white; font-size: 24px;">Create Account</h2>
            <p style="margin: 0 0 20px 0; color: rgba(255, 255, 255, 0.6); font-size: 14px;">
                Sign up to access the chatbot
            </p>
            <div style="background: rgba(255, 193, 7, 0.2); border: 1px solid rgba(255, 193, 7, 0.5); color: #ffb74d; padding: 12px; border-radius: 6px; margin-bottom: 16px; font-size: 12px;">
                <strong>Note:</strong> This is a development environment. Passwords are stored in plain text.
            </div>
            <form id="signup-form">
                <label for="signup-username">Username</label>
                <input type="text" id="signup-username" name="username" required minlength="3" autocomplete="off" />
                <div class="hint">At least 3 characters, no colons or commas</div>

                <label for="signup-password">Password</label>
                <input type="password" id="signup-password" name="password" required minlength="6" autocomplete="new-password" />
                <div class="hint">At least 6 characters, no colons or commas</div>

                <label for="signup-confirm">Confirm Password</label>
                <input type="password" id="signup-confirm" name="confirm" required autocomplete="new-password" />

                <div id="signup-message"></div>

                <div class="modal-buttons">
                    <button type="button" id="signup-cancel">Cancel</button>
                    <button type="submit" id="signup-submit">Create Account</button>
                </div>
            </form>
        `;

        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // Add event listeners
        document.getElementById('signup-cancel').onclick = () => modal.remove();
        modal.onclick = (e) => {
            if (e.target === modal) modal.remove();
        };

        document.getElementById('signup-form').onsubmit = async (e) => {
            e.preventDefault();
            await handleSignUp(modal);
        };

        // Focus username field
        document.getElementById('signup-username').focus();
    }

    async function handleSignUp(modal) {
        const username = document.getElementById('signup-username').value.trim();
        const password = document.getElementById('signup-password').value;
        const confirm = document.getElementById('signup-confirm').value;
        const messageDiv = document.getElementById('signup-message');
        const submitBtn = document.getElementById('signup-submit');

        // Clear previous messages
        messageDiv.innerHTML = '';

        // Validate passwords match
        if (password !== confirm) {
            messageDiv.innerHTML = '<div class="error-message">Passwords do not match</div>';
            return;
        }

        // Validate characters
        if (username.includes(':') || username.includes(',')) {
            messageDiv.innerHTML = '<div class="error-message">Username cannot contain : or ,</div>';
            return;
        }

        if (password.includes(':') || password.includes(',')) {
            messageDiv.innerHTML = '<div class="error-message">Password cannot contain : or ,</div>';
            return;
        }

        // Disable button
        submitBtn.disabled = true;
        submitBtn.textContent = 'Creating...';

        try {
            // Call the sign-up API
            const response = await fetch('/signup-api', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password })
            });

            const result = await response.json();

            if (result.success) {
                messageDiv.innerHTML = '<div class="success-message">' + result.message + ' Redirecting to login...</div>';

                // Close modal and refresh after short delay
                setTimeout(() => {
                    modal.remove();
                    // Reload page to get fresh login form
                    window.location.reload();
                }, 2000);
            } else {
                messageDiv.innerHTML = '<div class="error-message">' + result.message + '</div>';
                submitBtn.disabled = false;
                submitBtn.textContent = 'Create Account';
            }
        } catch (error) {
            messageDiv.innerHTML = '<div class="error-message">Error: ' + error.message + '</div>';
            submitBtn.disabled = false;
            submitBtn.textContent = 'Create Account';
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
