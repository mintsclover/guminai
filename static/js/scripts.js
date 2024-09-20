document.addEventListener('DOMContentLoaded', function () {
    // 기존 변수 선언
    const sendButton = document.getElementById('send-button');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const modelSelect = document.getElementById('model-select');
    const modelIntro = document.getElementById('model-intro');

    // 봇 아바타 이미지 소스 변수 추가
    let botAvatarSrc = '/static/images/bot_avatar.png';
    updateBotAvatar();

    // 모델 설명 및 봇 아바타 업데이트 함수
    function updateModelIntro() {
        const selectedModel = modelSelect.value;
        if (selectedModel === 'model1') {
            modelIntro.textContent = '환영합니다! 문서 기반으로 학습된 성내구민과 대화해보세요!';
        } else if (selectedModel === 'model2') {
            modelIntro.textContent = '환영합니다! 문서 기반으로 학습된 성동구민과 대화해보세요!';
        }
        // 모델에 따라 봇 아바타 이미지 변경
        updateBotAvatar();
    }

    // 봇 아바타 이미지 업데이트 함수
    function updateBotAvatar() {
        const selectedModel = modelSelect.value;
        if (selectedModel === 'model1') {
            botAvatarSrc = '/static/images/bot_avatar_model1.png';
        } else if (selectedModel === 'model2') {
            botAvatarSrc = '/static/images/bot_avatar_model2.png';
        } else {
            botAvatarSrc = '/static/images/bot_avatar.png'; // 기본 아바타
        }
    }

    // 초기 로드 시 모델 소개 업데이트
    updateModelIntro();

    modelSelect.addEventListener('change', function() {
        updateModelIntro();
    });

    sendButton.addEventListener('click', sendMessage);
    
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

    if (isMobile) {
        chatInput.addEventListener('keydown', function (e) {
            // 모바일에서는 Enter 키가 줄바꿈
            if (e.key === 'Enter' && !e.shiftKey) {
                // 기본 동작 허용 (줄바꿈)
            }
        });
    } else {
        chatInput.addEventListener('keydown', function (e) {
            // 데스크탑에서는 Enter 키로 전송
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    chatInput.addEventListener('input', function () {
        // 입력란의 높이를 자동으로 조절
        chatInput.style.height = 'auto';
        chatInput.style.height = chatInput.scrollHeight + 'px';
    });

    
    const exampleQuestions = document.querySelectorAll('.example-question');

    exampleQuestions.forEach(function (button) {
        button.addEventListener('click', function () {
            const question = this.textContent;
            chatInput.value = question;
            chatInput.style.height = 'auto';
            sendMessage();
        });
    });


    const toggleButton = document.getElementById('toggle-questions');
    const questionsContainer = document.getElementById('questions-container');

    toggleButton.addEventListener('click', function () {
        if (questionsContainer.style.display === 'none') {
            questionsContainer.style.display = 'flex';
            toggleButton.textContent = '접기 ▲';
        } else {
            questionsContainer.style.display = 'none';
            toggleButton.textContent = '펼치기 ▼';
        }
    });

    const menuButton = document.getElementById('menu-button');
    const sideMenu = document.getElementById('side-menu');
    const overlay = document.createElement('div');
    overlay.id = 'overlay';
    document.body.appendChild(overlay);

    menuButton.addEventListener('click', function () {
        sideMenu.classList.add('open');
        overlay.classList.add('show');
    });

    overlay.addEventListener('click', function () {
        sideMenu.classList.remove('open');
        overlay.classList.remove('show');
    });

    function sendMessage() {
        const message = chatInput.value.trim();
        if (message === '') return;

        addMessage('user', message);
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // 타이핑 인디케이터 추가
        showTypingIndicator();

        // 모델 소개 박스 숨기기
        if (modelIntro) {
            modelIntro.style.display = 'none';
        }

        // 모델 선택 드롭다운 비활성화
        modelSelect.disabled = true;
        sendButton.disabled = true;
        sendButton.classList.add('disabled'); // 비활성화 클래스 추가
        //sendButton.querySelector('img').src = '/static/images/send_icon_disabled.png';

        fetch('/chat_api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                model: modelSelect.value
            })
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            // 모델 선택 드롭다운 및 전송 버튼 다시 활성화
            modelSelect.disabled = false;
            sendButton.disabled = false;
            sendButton.classList.remove('disabled'); // 비활성화 클래스 제거
            //sendButton.querySelector('img').src = '/static/images/send_icon.png';
            if (data.answer) {
                addMessage('bot', data.answer);
            } else {
                addMessage('bot', '죄송합니다. 응답을 가져올 수 없습니다.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            hideTypingIndicator();
            // 모델 선택 드롭다운 다시 활성화
            modelSelect.disabled = false;
            sendButton.disabled = false;
            sendButton.classList.remove('disabled');
            //sendButton.querySelector('img').src = '/static/images/send_icon.png';
            addMessage('bot', '죄송합니다. 오류가 발생했습니다.');
        });
    }

    function addMessage(sender, text) {
        if (sender === 'system') {
            const messageElement = document.createElement('div');
            messageElement.classList.add('system-message');
            messageElement.textContent = text;
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            return;
        }

        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);

        const avatar = document.createElement('img');
        avatar.classList.add('avatar');
        if (sender === 'user') {
            avatar.src = '/static/images/user_avatar.png';
        } else {
            avatar.src = botAvatarSrc; // 선택된 모델에 따른 봇 아바타 이미지 사용
        }

        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');

        if (sender === 'bot') {
            typeText(messageContent, text);
        } else {
            messageContent.textContent = text;
        }

        if (sender === 'user') {
            messageElement.appendChild(messageContent);
            messageElement.appendChild(avatar);
        } else {
            messageElement.appendChild(avatar);
            messageElement.appendChild(messageContent);
        }

        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function typeText(element, text, index = 0) {
        if (index < text.length) {
            element.textContent += text.charAt(index);
            setTimeout(() => {
                typeText(element, text, index + 1);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 30); // 타이핑 속도 조절 (밀리초)
        }
    }

    function showTypingIndicator() {
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('message', 'bot', 'typing-indicator');

        const avatar = document.createElement('img');
        avatar.classList.add('avatar');
        avatar.src = botAvatarSrc; // 선택된 모델에 따른 봇 아바타 이미지 사용

        const dots = document.createElement('div');
        dots.classList.add('message-content');
        dots.innerHTML = `
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
        `;

        typingIndicator.appendChild(avatar);
        typingIndicator.appendChild(dots);

        typingIndicator.id = 'typing-indicator';
        chatMessages.appendChild(typingIndicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            chatMessages.removeChild(typingIndicator);
        }
    }
});